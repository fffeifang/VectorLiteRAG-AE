import os
import time
import faiss
import queue
import threading
import numpy as np
import numba as nb

from pathlib import Path
from typing import Callable
from collections import deque
from multiprocessing import Queue
from vliterag.configs import vLiteConfigs
from concurrent.futures import ThreadPoolExecutor
from faiss.contrib.ivf_tools import replace_ivf_quantizer

REFRESH_INTERVAL = 1000  # refresh cluster access stats every N queries

@nb.njit(cache=True)
def _ranks_and_group(key, sid_m, nq, nshards):
    """
    key    : sid * nq + qid      (int32)
    sid_m  : shard id in [0, nshards-1]  (int32)
    returns:
    ranks         : rank within (sid,qid) group, size N
    idx_grouped   : permutation of [0..N), grouped by shard (all sid=0, then sid=1, ...)
    shard_starts  : starts (size nshards+1) for slices in idx_grouped
    widths        : max count per qid for each shard (size nshards)
    """
    N = key.size
    K = nshards * nq

    # counts per (sid,qid)
    key_ctr = np.zeros(K, dtype=np.int32)

    # totals per shard to build a contiguous grouping by shard
    shard_totals = np.zeros(nshards, dtype=np.int64)
    for i in range(N):
        shard_totals[sid_m[i]] += 1

    shard_starts = np.empty(nshards + 1, dtype=np.int64)
    shard_starts[0] = 0
    for s in range(nshards):
        shard_starts[s + 1] = shard_starts[s] + shard_totals[s]

    shard_next = shard_starts[:-1].copy()

    ranks = np.empty(N, dtype=np.int32)
    idx_grouped = np.empty(N, dtype=np.int64)

    # single pass: compute rank within key and group by shard
    for i in range(N):
        k = key[i]
        ranks[i] = key_ctr[k]
        key_ctr[k] += 1

        s = sid_m[i]
        pos = shard_next[s]
        idx_grouped[pos] = i
        shard_next[s] = pos + 1

    # widths per shard = max over qid of counts[sid,qid]
    widths = np.zeros(nshards, dtype=np.int32)
    # reshape view of counts
    # (safe because K = nshards * nq)
    counts_2d = key_ctr.reshape(nshards, nq)
    for s in range(nshards):
        m = 0
        row = counts_2d[s]
        for q in range(nq):
            c = row[q]
            if c > m:
                m = c
        widths[s] = m

    return ranks, idx_grouped, shard_starts, widths

class BaseIndex:
    def __init__(self, cfgs, nprobe=2048):
        self.cfgs = cfgs
        self.nprobe = nprobe
        self.index = None
        self.search_param = self._set_search_parameters()
        self.index_dir = Path(cfgs.database_dir) / cfgs.index

    def init_index(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def search(self, queries, k):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def register_callback(self, callback: Callable):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def reset_counters(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def _set_search_parameters(self):
        return faiss.SearchParametersIVF(
            nprobe=self.nprobe,
            quantizer_params=faiss.SearchParametersHNSW(
                efSearch=int(self.nprobe * 1.1)
            )
        )
    
    def get_shards_dir(self):
        return self.index_dir /  self.cfgs.model / f"{self.cfgs.num_gpus}gpus"

class BaseGPUIndex(BaseIndex):
    def __init__(self, cfgs, nprobe=2048):
        super().__init__(cfgs, nprobe)
        self.slo = cfgs.search_slo

    def init_index(self, verbose=False, ef_construction=200):
        ivfpq_path = self.index_dir / 'ivfpq.index'
        if not ivfpq_path.exists():
            raise FileNotFoundError(f"Index file not found: {ivfpq_path}")
        self.index = faiss.read_index(str(ivfpq_path))
        if not isinstance(faiss.downcast_index(self.index), faiss.IndexIVFPQ):
            raise TypeError("The index must be an IVFPQ index.")
        
        print(f"index ntotal: {self.index.ntotal}")
        
        quantizer = faiss.downcast_index(self.index.quantizer)
        if self.cfgs.search_mode == 'ded-gpu':
            index_ivf = faiss.extract_index_ivf(self.index)
            new_quantizer = faiss.IndexFlatL2(index_ivf.d)
            replace_ivf_quantizer(index_ivf, new_quantizer)
            print(f"[VLITE] Replaced quantizer with IndexFlatL2 for IVF index.")
        elif self.cfgs.search_mode == 'all-gpu': 
            if not isinstance(quantizer, faiss.IndexHNSWFlat):
                index_ivf = faiss.extract_index_ivf(self.index)
                new_quantizer = faiss.IndexHNSWFlat(index_ivf.d, 32, faiss.METRIC_L2)
                new_quantizer.hnsw.efConstruction = ef_construction
                new_quantizer.hnsw.efSearch = int(self.nprobe * 1.5)
                replace_ivf_quantizer(index_ivf, new_quantizer)
                print(f"[VLITE] Replaced quantizer with IndexHNSWFlat for IVF index.")
        elif isinstance(quantizer, faiss.IndexHNSWFlat):
            quantizer.hnsw.efSearch = int(self.nprobe * 1.5)
            print(f"[VLITE] Using existing IndexHNSWFlat quantizer with efSearch={quantizer.hnsw.efSearch}.")

        co = faiss.GpuMultipleClonerOptions()
        co.verbose = True
        co.useFloat16 = True
        co.usePrecomputed = False
        co.common_ivf_quantizer = True
        co.allowCpuCoarseQuantizer = True
        co.indicesOptions = faiss.INDICES_32_BIT
        
        if self.cfgs.search_mode == 'ded-gpu':
            co.useFloat16CoarseQuantizer = True
            co.allowCpuCoarseQuantizer = False
        
        if self.cfgs.search_mode == 'all-gpu':  # all-gpu mode
            co.shard = True
            self.gpu_index = faiss.index_cpu_to_all_gpus(
                self.index, co, self.cfgs.num_gpus)
            sharded_index = faiss.downcast_index(self.gpu_index)
            for rank in range(self.cfgs.num_gpus):
                gpu_index = faiss.downcast_index(sharded_index.at(rank))
                gpu_index.nprobe = self.nprobe
        elif self.cfgs.search_mode == 'ded-gpu':  # dedicated GPU mode
            if self.cfgs.ann_workers == 1:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfgs.num_gpus - 1) # last GPU for dedicated mode
                co.shard = False
                res = faiss.StandardGpuResources()
                self.gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index, co)
                ivf_index = faiss.downcast_index(self.gpu_index)
                ivf_index.nprobe = self.nprobe
            else:
                print(f"[VLITE] Using {self.cfgs.ann_workers} dedicated ANN workers.")
                co.shard = True
                devices = list(str(no) for no in range(self.cfgs.num_gpus - self.cfgs.ann_workers, self.cfgs.num_gpus))
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
                self.gpu_index = faiss.index_cpu_to_all_gpus(self.index, co, self.cfgs.ann_workers)
                sharded_index = faiss.downcast_index(self.gpu_index)
                for rank in range(self.cfgs.ann_workers):
                    gpu_index = faiss.downcast_index(sharded_index.at(rank))
                    gpu_index.nprobe = self.nprobe

    def search(self, queries, k):
        # todo: dedicated base line may use multiple GPUs
        return self.gpu_index.search(queries, k, self.search_param)
    
    def register_callback(self, callback):
        return  # No callback for GPU index, as it is not used in the current implementation.
    
    def reset_counters(self):
        return

class ShardedIndex(BaseIndex):
    def __init__(self, cfgs, nprobe=2048):
        super().__init__(cfgs, nprobe)
    
    def init_index(self, verbose=False):
        if self.cfgs.search_mode == 'hedrarag':
            ipqfs_path = self.index_dir / "hedrarag_ivffs.index"
        else:
            ipqfs_path = self.index_dir / "ivffs.index"
            
        if not ipqfs_path.exists():
            raise FileNotFoundError(f"Fastscan Index file not found: {ipqfs_path}")
        self.index = faiss.read_index(str(ipqfs_path))
        print("[VLITE] FastScan Index loaded from", ipqfs_path)
        self.index.implem = 12  # fastest implementation
        self.index.use_precomputed_table = True
        self.index.collect_breakdown = True
        self.index.verbose_dispatch = False
        
        if self.cfgs.search_mode:
            if self.cfgs.search_mode == 'hedrarag':
                gpuivf_path = str(self.get_shards_dir() / f'hedrarag_{self.nprobe}.index')
                maptab_path = str(self.get_shards_dir() / f'hedrarag_{self.nprobe}.imap')
            else:
                gpuivf_path = str(self.get_shards_dir() / f'{self.cfgs.search_slo}_{self.nprobe}.index')
                maptab_path = str(self.get_shards_dir() / f'{self.cfgs.search_slo}_{self.nprobe}.imap')                
            self.index.set_gpu_index(gpuivf_path, maptab_path, True)    # collect breakdown = True
        elif self.cfgs.search_slo < 0: 
            raise ValueError(f"Invalid SLO value: {self.cfgs.search_slo}. Expected a non-negative integer for FastScan index.")

    def register_callback(self, callback: Callable):
        if self.cfgs.dispatcher:
            print("[VLITE] Dispatcher enabled")
            self.index.register_callback(callback, False)
            
    def reset_counters(self):
        self.index.reset_counters()
        
    def search(self, queries, k):
        return self.index.search(queries, k, self.search_param)
    
    def _set_num_threads(self):
        faiss.omp_set_num_threads(faiss.get_num_threads())
        
class PartitionedIndex(BaseIndex):
    def __init__(self, cfgs, nprobe=2048, outQueue: Queue = None):
        super().__init__(cfgs, nprobe)
        self.quantizer = None
        self.sharded_indexes = []
        self.cid_lut = None
        self.shard_lut = None
        
        self.cluster_access = None
        self.num_slo_attained = 0

        self.request_no = 0
        self.queue = queue.Queue()
        self.gpuResults = [None] * self.cfgs.num_gpus
        self.gpuDoneFlags = [threading.Event() for _ in range(cfgs.num_gpus)]
        self.outQueue = outQueue

    def init_index(self, verbose=False):
        if self.cfgs.search_slo < 0:
            raise ValueError(f"Invalid SLO value: {self.cfgs.search_slo}. Expected a non-negative integer for Sharded index.")
        
        for flag in self.gpuDoneFlags:
            flag.clear()
        
        ipqfs_path = self.index_dir / 'ivffs.index'
        if not ipqfs_path.exists():
            raise FileNotFoundError(f"Fastscan Index file not found: {ipqfs_path}")
        shard_paths = self.get_shards_paths()
        assert(len(shard_paths) == self.cfgs.num_gpus), \
            f"Number of shard paths {len(shard_paths)} does not match number of GPUs {self.cfgs.num_gpus}."
        
        self.index = faiss.read_index(str(ipqfs_path))
        print("[VLITE] FastScan Index loaded from", ipqfs_path)
        self.index.implem = 12
        self.index.use_precomputed_table = True
        self.index.collect_breakdown = False
        self.index.verbose_dispatch = verbose
        self.quantizer = faiss.downcast_index(self.index.quantizer)
        self.cluster_access = np.zeros(self.quantizer.ntotal, dtype=np.int32)
        
        t0 = time.time()
        for i in range(self.cfgs.num_gpus):
            shard_path = shard_paths[i]
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard index file not found: {shard_path}")
            shard_index_cpu = faiss.read_index(str(shard_path))
            
            co = faiss.GpuClonerOptions()
            co.verbose = True
            co.useFloat16 = True
            co.usePrecomputed = True
            co.allowCpuCoarseQuantizer = True
            co.indicesOptions = faiss.INDICES_32_BIT
            
            t1 = time.time()
            shard_index_gpu = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), i, shard_index_cpu, co)
            print(f"[VLITE] Shard {i} loaded to GPU {i} in {time.time() - t1:.2f}s")
            self.sharded_indexes.append(shard_index_gpu)

        self._load_mapping_table()
        print(f"[VLITE] All {self.cfgs.num_gpus} shards and mapping tables loaded in {time.time() - t0:.2f}s")
    
    def quantize(self, queries):
        search_param = faiss.SearchParametersHNSW(
            efSearch=int(self.nprobe * 1.1))
        if not isinstance(self.quantizer, faiss.IndexHNSWFlat):
            raise TypeError("Quantizer must be an IndexHNSWFlat.")
        Dq, Iq = self.quantizer.search(queries, self.nprobe, search_param)
        np.add.at(self.cluster_access, Iq.reshape(-1), 1)
        return Dq, Iq

    def route_queries(self, Dq, Iq):
        nq, nprobe = Dq.shape
        N = nq * nprobe

        flat_qids = np.repeat(np.arange(nq, dtype=np.int32), nprobe)
        flat_cids = Iq.reshape(-1)
        flat_dist = Dq.reshape(-1).astype(np.float32, copy=False)

        nshards = self.cfgs.num_gpus + 1
        shard_ids = self.shard_lut[flat_cids]
        shard_ids_m = np.where(shard_ids == -1, nshards - 1, shard_ids).astype(np.int32, copy=False)
        new_cids = self.cid_lut[flat_cids].astype(np.int32, copy=False)

        key = (shard_ids_m.astype(np.int64) * nq + flat_qids).astype(np.int32, copy=False)
        ranks, idx_grouped, shard_starts, widths = _ranks_and_group(key, shard_ids_m, nq, nshards)

        Dq_dict, Iq_dict = {}, {}

        for sid_m in range(nshards):
            w = int(widths[sid_m])
            out_key = sid_m if sid_m < nshards - 1 else -1

            if w == 0:
                Dq_dict[out_key] = np.full((nq, 0), np.inf, dtype=np.float32)
                Iq_dict[out_key] = np.full((nq, 0), -1, dtype=np.int32)
                continue

            D_out = np.full((nq, w), np.inf, dtype=np.float32)
            I_out = np.full((nq, w), -1, dtype=np.int32)

            lo = int(shard_starts[sid_m])
            hi = int(shard_starts[sid_m + 1])
            if lo < hi:
                idxs = idx_grouped[lo:hi]
                rows = flat_qids[idxs]
                cols = ranks[idxs].astype(np.int64, copy=False)
                D_out[rows, cols] = flat_dist[idxs]
                I_out[rows, cols] = new_cids[idxs]

            Dq_dict[out_key] = D_out
            Iq_dict[out_key] = I_out

        return Dq_dict, Iq_dict

    def merge_and_rerank(self, Dcpu, Icpu, Dgpu_dict, Igpu_dict):
        ts_0 = time.time()
        nq, k = Dcpu.shape
        Dmerged = np.zeros((nq, k), dtype=np.float32)
        Imerged = np.zeros((nq, k), dtype=np.int32)
        
        if self.cfgs.dispatcher:
            return Dmerged, Imerged

        def merge_one(i):
            ts_0 = time.time()
            d_row = [Dcpu[i]]
            i_row = [Icpu[i]]

            for g in Dgpu_dict:
                if Dgpu_dict[g] is not None:
                    d_row.append(Dgpu_dict[g][i])
                    i_row.append(Igpu_dict[g][i])

            all_distances = np.concatenate(d_row)
            all_indices = np.concatenate(i_row)

            topk_idx = np.argsort(all_distances)[:k]
            return i, all_distances[topk_idx], all_indices[topk_idx]

        with ThreadPoolExecutor() as executor:
            for i, d_topk, i_topk in executor.map(merge_one, range(nq)):
                Dmerged[i] = d_topk
                Imerged[i] = i_topk
        return Dmerged, Imerged

    def search(self, queries, k):
        ts_0 = time.time()
        Dq, Iq = self.quantize(queries)
        ts_1 = time.time()
        Dq_dict, Iq_dict = self.route_queries(Dq, Iq)
        ts_2 = time.time()
        
        cpu_Iq = Iq_dict[-1]
        cpu_Dq = Dq_dict[-1]
        gpu_nprobes = {g: Iq_dict[g].shape[1] for g in Iq_dict}
        miss_nprobe =  np.sum(cpu_Iq != -1, axis=1)

        Dgpu_dict = {}
        Igpu_dict = {}
        
        if self.cfgs.dispatcher:
            dispatcher_thread = threading.Thread(
                target=self.dispatcher, args=(cpu_Iq, k))
            dispatcher_thread.start()
        
        gpu_threads = []
        for gid, gpu_index in enumerate(self.sharded_indexes):
            gpu_index.nprobe = gpu_nprobes[gid]
            if gpu_nprobes[gid] == 0:
                self.gpuDoneFlags[gid].set()
                continue
            
            def gpu_worker(gpu_id, queries, k, iq, dq, igpu, dgpu):
                D, I = gpu_index.search_preassigned(
                    queries, k, iq[gpu_id], dq[gpu_id])
                dgpu[gpu_id] = D
                igpu[gpu_id] = I
                self.gpuResults[gpu_id] = (D, I)
                self.gpuDoneFlags[gpu_id].set()

            gpu_thread = threading.Thread(
                target=gpu_worker, 
                args=(gid, queries, k, Iq_dict, Dq_dict, Dgpu_dict, Igpu_dict))
            gpu_thread.start()
            gpu_threads.append(gpu_thread)

        if cpu_Iq.shape[1] > 0:
            self.index.nprobe = cpu_Iq.shape[1]
            Dcpu, Icpu = self.index.search_preassigned(queries, k, cpu_Iq, cpu_Dq)
        else:
            Dcpu, Icpu = np.full((queries.shape[0], k), np.inf, dtype=np.float32), \
                            np.full((queries.shape[0], k), -1, dtype=np.int32)
            
        for thread in gpu_threads:
            thread.join()
            
        if self.cfgs.dispatcher:
            dispatcher_thread.join()

        ts_3 = time.time()
        D, I = self.merge_and_rerank(Dcpu, Icpu, Dgpu_dict, Igpu_dict)
        ts_4 = time.time()
        
        # for stats
        D[0,0] = ts_1 - ts_0
        D[:,9] = miss_nprobe
        
        batch_size = D.shape[0]
        slo_attn = self._update_counters(D, ts_4 - ts_0)
        # print(f"[VLITE] Quant Time: {ts_1 - ts_0:.3f}s, Route Time: {ts_2 - ts_1:.3f}s, Dispatch Time: {ts_3 - ts_2:.3f}s, Merge Time: {ts_4 - ts_3:.3f}s")
        
        if 0 <= slo_attn < 0.9:
            print(f"[VLITE] Average SLO attainment in last {REFRESH_INTERVAL} queries: {slo_attn:.3f}")
            threading.Thread(
                target=self._repartition_clusters, 
                args=(self.cluster_access.copy(),)).start()
            self.cluster_access.fill(0)

        for flag in self.gpuDoneFlags:
            flag.clear()
            
        if self.queue.qsize() > 0:
            print(f"[VLITE] Warning: Queue is not empty after search. Size: {self.queue.qsize()}")            
        return D, I

    def dispatcher(self, cpu_Iq, k):
        n = cpu_Iq.shape[0]
        ts_0 = time.time()
        
        for flag in self.gpuDoneFlags:
            flag.wait()
            
        nshards = self.cfgs.num_gpus + 1
        
        Ds = np.full((nshards, n, k), np.inf, dtype=np.float32)
        Is = np.full((nshards, n, k), -1, dtype=np.int32)
        
        for gid in range(self.cfgs.num_gpus):
            result = self.gpuResults[gid]
            if result is not None:
                D, I = result
                Ds[gid] = D
                Is[gid] = I
                self.gpuResults[gid] = None
        
        gpu_only_lids = deque(np.flatnonzero((cpu_Iq == -1).all(axis=1)))
        processed = np.zeros(n, dtype=bool)
        processed_cnt = 0
        
        while processed_cnt < n:
            if gpu_only_lids:
                lid = int(gpu_only_lids.popleft())
            else:
                try:
                    _, lid, ids, dis = self.queue.get(timeout=0.01)
                except queue.Empty:
                    continue
                
                Is[-1, lid, :] = ids
                Ds[-1, lid, :] = dis

            topk_dis, topk_ids = self._merge_topk(Ds[:, lid, :], Is[:, lid, :], k)
            rid = self.request_no + lid
            tsCallback = time.time()
            # print(f"[VLITE] Dispatching request {rid} for local ID {lid+1}/{n} - time: {tsCallback - ts_0:.3f}s")
            self.outQueue.put((n, rid, topk_ids, tsCallback))
            
            if not processed[lid]:
                processed[lid] = True
                processed_cnt += 1

    def _merge_topk(self, D_row, I_row, k):
        m_dis = D_row.reshape(-1)
        m_ids = I_row.reshape(-1)
        kt = k - 1
        idx_k = np.argpartition(m_dis, kt)[:k]
        order = np.argsort(m_dis[idx_k], kind='stable')
        idx = idx_k[order]
        
        return m_dis[idx], m_ids[idx]

    def register_callback(self, dummy_callback):
        def callback(request_id, local_id, k, ids, dis):
            self.queue.put((request_id, local_id, ids, dis))
        
        if self.cfgs.dispatcher:
            print("[VLITE] Dispatcher enabled")
            self.index.register_callback(callback, False)
            
    def reset_counters(self):
        self.index.reset_counters()
        self.request_no = 0
        self.cluster_access.fill(0)
    
    def _load_mapping_table(self):
        mtab_path = self.get_shards_dir() / f"{self.cfgs.search_slo}_{self.nprobe}.imap"
        if not mtab_path.exists():
            raise FileNotFoundError(f"Mapping table file not found: {mtab_path}")
        data = np.fromfile(mtab_path, dtype=np.int32).reshape(-1, 3)
        
        nlist = self.index.nlist 
        cid_lut = np.full(nlist, -1, dtype=np.int32)
        shard_lut = np.full(nlist, -1, dtype=np.int16)
        
        for orig, shard, new in data:
            if orig < 0 or orig >= nlist:
                raise ValueError(f"Invalid original CID {orig} in mapping table.")
            if shard < 0 or shard >= self.cfgs.num_gpus:
                raise ValueError(f"Invalid shard ID {shard} in mapping table.")
            if new < 0 or new >= nlist:
                raise ValueError(f"Invalid new CID {new} in mapping table.")
            cid_lut[orig] = new
            shard_lut[orig] = shard
        
        for cid in range(nlist):
            if cid_lut[cid] == -1:
                cid_lut[cid] = cid

        self.cid_lut = cid_lut
        self.shard_lut = shard_lut
        
    def _update_counters(self, D, latency):
        batch_size = D.shape[0]
        self.request_no += batch_size
        self.num_slo_attained += batch_size if latency <= self.cfgs.search_slo else 0
        
        if self.request_no > 0 and self.request_no % REFRESH_INTERVAL == 0:
            avg_slo_attainment = self.num_slo_attained / REFRESH_INTERVAL
            self.num_slo_attained = 0
            return avg_slo_attainment
        return -1

    def _repartition_clusters(self, access_counts):
        pass
    
    def get_shards_paths(self):
        base = self.index_dir /  self.cfgs.model / f"{self.cfgs.num_gpus}gpus"
        path_list = [(base / f'{self.cfgs.search_slo}_{i}_{self.nprobe}.index') for i in range(self.cfgs.num_gpus)]
        return path_list
        
        