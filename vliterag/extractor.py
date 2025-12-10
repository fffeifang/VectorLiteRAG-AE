import os
import tqdm
import time
import faiss
import numpy as np

from pathlib import Path
from configs.loader import load_index
from vliterag.args import parse_args
from vliterag.utils import load_vectors
from vliterag.configs import vLiteConfigs

from faiss.contrib.inspect_tools import get_invlist_sizes, get_invlist
from faiss.contrib.ivf_tools import replace_ivf_quantizer

class IndexSplitter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.index = None
        self.index_dir = Path()
        self.shard_dir = Path()
        self.set_dirs(cfg)
    
    def set_dirs(self, cfg):
        self.index_dir = Path(cfg.database_dir) / cfg.index
        self.shard_dir = Path(self.index_dir) / cfg.model / f"{cfg.num_gpus}gpus"
        
    def get_orig_index(self):
        if self.index is None:
            if self.cfg.search_mode == 'hedrarag':
                index_name = f'hedrarag_ivfpq.index'
            else:
                index_name = f"ivfpq.index"
                
            index_path = Path(self.cfg.database_dir) / self.cfg.index / index_name
            self.index = faiss.clone_index(faiss.read_index(str(index_path)))
            print(f"[VLITE] Read original IVF index from {index_path}")

    def get_partitioned_cids(self):
        centroid_dir = Path(self.shard_dir) / "shards"
        nprobe = self.cfg.search_nprobe
        if self.cfg.search_mode == 'hedrarag':
            ids_path = centroid_dir / f"hedrarag_{nprobe}.index"
        else:
            ids_path = centroid_dir / f"{self.cfg.search_slo}ms_cids_{nprobe}.npz"
        
        if ids_path.exists():
            return np.load(ids_path, allow_pickle=True)['ids']
        else:
            return None
    
    def save_mapping_table(self, mapping_table):
        nprobe = self.cfg.search_nprobe
        if self.cfg.search_mode == 'hedrarag':
            mtab_path = self.shard_dir / f'hedrarag_{nprobe}.imap'
        else:
            mtab_path = self.shard_dir / f"{self.cfg.search_slo}_{nprobe}.imap"
            
        items = list(mapping_table.items())
        num_items = len(items)
        
        with open(mtab_path, 'wb') as f:
            f.write(np.int32(num_items).tobytes())
            for cid, new_cid in items:
                f.write(np.int32(cid).tobytes())
                f.write(np.int32(new_cid).tobytes())
        print(f"[VLITE] Saved mapping table to {mtab_path}")
        
    def save_mapping_table_group(self, mapping_table):
        nprobe = self.cfg.search_nprobe
        mtab_path = self.shard_dir / f"{self.cfg.search_slo}_{nprobe}.imap"
            
        arr = np.array([(cid, shard, newcid) for cid, (shard, newcid) in mapping_table.items()], dtype=np.int32)
        arr.tofile(mtab_path)
        print(f"[VLITE] Saved mapping table to {mtab_path}, {arr.shape}")
    
    def shard_ivf(self, partitioned_cids):
        orig_ivfpq = self.index
        orig_invlists = orig_ivfpq.invlists
        list_sizes = get_invlist_sizes(orig_invlists)        
        
        if len(partitioned_cids) == orig_ivfpq.nlist:
            return
            
        pp_list_sizes = np.zeros(len(partitioned_cids), dtype=np.int32)
        for i, cid in enumerate(partitioned_cids):
            if cid == -1:
                pp_list_sizes[i] = -1
            else:
                pp_list_sizes[i] = list_sizes[int(cid)]
        sorted_indices = np.argsort(pp_list_sizes)
        sorted_partitioned_cids = np.array(partitioned_cids)[sorted_indices]
        pp_list_sizes = pp_list_sizes[sorted_indices]

        pp_nlist = len(sorted_partitioned_cids)
        pp_cq = faiss.IndexFlatL2(orig_ivfpq.d)
        pp_ivf = faiss.IndexIVFPQ(
            pp_cq, 
            orig_ivfpq.d, 
            pp_nlist, 
            orig_ivfpq.pq.M, 
            orig_ivfpq.pq.nbits, 
            faiss.METRIC_L2
        )
        
        pp_ivf.by_residual = True
        pp_ivf.is_trained = True
        pp_cq.is_trained = True
        pp_ivf.pq = orig_ivfpq.pq   # Caution copying the PQ object

        mapping_table = dict()
        # Add the partitioned centroids to the new IVF index
        for i in tqdm.tqdm(range(pp_nlist), desc="[VLITE] Adding centroids to new partitioned IVF index"):
            cid = int(sorted_partitioned_cids[i])
            new_cid = int(i)
            list_size = int(pp_list_sizes[new_cid])
            if cid == -1:
                continue
            
            list_ids, list_codes = get_invlist(orig_invlists, cid)
            pp_ivf.invlists.add_entries(
                new_cid,
                list_size,
                faiss.swig_ptr(list_ids),
                faiss.swig_ptr(list_codes),
            )
            pp_ivf.ntotal += int(pp_list_sizes[new_cid])    # cid's original size
            centroid_vector = orig_ivfpq.quantizer.reconstruct(cid)
            pp_cq.add(centroid_vector.reshape(1, -1))
            mapping_table[cid] = new_cid
        
        # Write the new IVF index to disk
        pp_ivf_dir = self.shard_dir / 'shards'
        pp_ivf_dir.mkdir(parents=True, exist_ok=True)
        pp_ivf_path = pp_ivf_dir / 'ppivf.index'
            
        faiss.write_index(pp_ivf, str(pp_ivf_path))
        print(f"[VLITE] Sharded IVF index saved to {pp_ivf_path}")
        
        self.save_mapping_table(mapping_table)

    def test_ivfpq(self):
        pp_ivf_dir = self.shard_dir / 'shards'
        pp_ivf_dir.mkdir(parents=True, exist_ok=True)
        pp_ivf_path = pp_ivf_dir / 'ppivf.index'
        
        pp_ivfpq = faiss.read_index(pp_ivf_path)
        is_trained = pp_ivfpq.is_trained
        cq_is_trained = pp_ivfpq.quantizer.is_trained
        pq_M = pp_ivfpq.pq.M
        pq_nbits = pp_ivfpq.pq.nbits
        pq_centroids = pp_ivfpq.pq.centroids.size()
        d = pp_ivfpq.d
        nlist = pp_ivfpq.nlist
        ntotal = pp_ivfpq.ntotal
        print(f"[VLITE] Partitioned IVF index: is_trained={is_trained}, cq_is_trained={cq_is_trained}, pq_centroids={pq_centroids}, "
              f"d={d}, nlist={nlist}, ntotal={ntotal}, M={pq_M}, nbits={pq_nbits}")

    def partition_ivf(self, partitioned_cids):
        num_shards = self.cfg.num_gpus
        orig_ivfpq = self.index
        orig_invlists = orig_ivfpq.invlists
        list_sizes = get_invlist_sizes(orig_invlists)        
        
        # this has redundancy: quantizers are copied. Maybe too much?
        if len(partitioned_cids) == orig_ivfpq.nlist:
            print("[VLITE] No partitioning needed.")
            return

        list_sizes = np.array([list_sizes[int(cid)] if cid != -1 else -1 for cid in partitioned_cids], dtype=np.int32)
        order_by_size = np.argsort(list_sizes)
        list_sizes = list_sizes[order_by_size]
        sorted_cids = np.array(partitioned_cids)[order_by_size]
        
        sharded_cids = []
        sharded_lsizes = []
        for i in range(num_shards):
            sharded_cids.append(sorted_cids[np.arange(len(sorted_cids)) % num_shards == i])
            sharded_lsizes.append(list_sizes[np.arange(len(list_sizes)) % num_shards == i])
        
        mapping_table = dict() # orig_cid -> (shard_id, new_cid)
        for i in range(num_shards):
            nlist = len(sharded_cids[i])
            quantizer = faiss.IndexFlatL2(orig_ivfpq.d)
            ivfpq = faiss.IndexIVFPQ(
                quantizer,
                orig_ivfpq.d,
                nlist,
                orig_ivfpq.pq.M,
                orig_ivfpq.pq.nbits,
                faiss.METRIC_L2
            )
            
            ivfpq.by_residual = True
            ivfpq.is_trained = True
            quantizer.is_trained = True
            ivfpq.pq = orig_ivfpq.pq
            
            for j in tqdm.tqdm(range(nlist), desc=f"[VLITE] Adding Centroids to new IVFPQ index ({i+1}/{num_shards} Shard)"):
                cid = int(sharded_cids[i][j])
                new_cid = int(j)
                list_size = int(sharded_lsizes[i][new_cid])
                if cid == -1:
                    continue
                
                list_ids, list_codes = get_invlist(orig_invlists, cid)
                ivfpq.invlists.add_entries(
                    new_cid,
                    list_size,
                    faiss.swig_ptr(list_ids),
                    faiss.swig_ptr(list_codes),
                )
                ivfpq.ntotal += int(sharded_lsizes[i][new_cid])
                centroid = orig_ivfpq.quantizer.reconstruct(cid)
                quantizer.add(centroid.reshape(1, -1))
                mapping_table[cid] = (i, new_cid)

            self.shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = self.shard_dir / f"{self.cfg.search_slo}_{i}_{self.cfg.search_nprobe}.index"
                
            faiss.write_index(ivfpq, str(shard_path))
            print(f"[VLITE] Partitioned IVF index saved to {shard_path}")
        
        self.save_mapping_table_group(mapping_table)

def extract(cfg):
    splitter = IndexSplitter(cfg)
    splitter.get_orig_index()
    
    if cfg.search_slo <= 0 or cfg.sweep:
        cfg.searh_slo = load_index()[cfg.index]['slo']
        
    t1 = time.time()

    cfg.set_result_paths()
    splitter.set_dirs(cfg)
        
    print(f"[VLITE] Splitting {cfg.index} index with slo={cfg.search_slo} for model {cfg.model}")
        
    partitioned_cids = splitter.get_partitioned_cids()
    if partitioned_cids is None:
        return
        
    if cfg.search_mode == 'hedrarag':
        splitter.shard_ivf(partitioned_cids)
    else:
        splitter.partition_ivf(partitioned_cids)
            
    print(f"[VLITE] Time taken for extracting IVF index: {time.time() - t1:.2f} seconds")