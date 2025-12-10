import os
import re
import time
import pwlf
import copy
import faiss
import torch
import asyncio
import numpy as np
import pandas as pd
from copy import deepcopy

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

from vliterag.args import parse_args
from vliterag.runner import run_pipeline
from vliterag.configs import vLiteConfigs
from vliterag.utils import prepare_queries
from vliterag.results import vLiteResults

from configs.loader import load_gpu, load_index, write_json

from index.memory_calculator import IndexMemoryCalculator
from index.index_wrapper import ShardedIndex

@dataclass
class LatencyModel:
    cpu_num: int
    search: List[Tuple[float, float, float, float]] # start, end, slope, intercept
    quantizer: List[Tuple[float, float, float, float]] # start, end, slope, intercept
    lut: List[Tuple[float, float, float, float]] # start, end, slope, intercept

    def add_data(self, attr, start, end, val1, val2):
        mval = self.__dict__[attr]
        mval.append((start, end, val1, val2))

class LatencyEstimator:
    def __init__(self, num_cpu, nprobe, 
                 latency_data : Optional[pd.DataFrame] = None,
                 hitrate_data : Optional[np.array] = None):
        self.nc = num_cpu
        self.nprobe = nprobe
        self.latency_model = None
        self.latency_model_gpu = None
        self.latency_data = latency_data
        self.latency_data_gpu = None
        self.hitrate_data = hitrate_data
        self.latency_margin = 1.0
        
        if self.latency_data is None:
            self.latency_data = pd.DataFrame(columns=[
                'Batch Size', 'Search', 'Quantization',
                'Compute LUT', 'Scan LUT', 'Hit Rate',
            ])

    def save_latency_model(self, cfgs):
        cpu_rgmodel_file = Path(cfgs.result_dir).parent / f"latency_regression_cpu_{self.nprobe}.csv"
        gpu_rgmodel_file = Path(cfgs.result_dir).parent / f"latency_regression_gpu_{self.nprobe}.csv"

        if self.latency_model is None:
            return
        
        with open(cpu_rgmodel_file, 'w') as f:
            f.write("sstart,send,sslope,ssintpt,qstart,qend,qslope,qsintpt,lstart,lend,lslope,lsintpt\n")
            for i in range(len(self.latency_model.search)):
                sstart, send, sslope, ssintpt = self.latency_model.search[i]
                qstart, qend, qslope, qsintpt = self.latency_model.quantizer[i]
                lstart, lend, lslope, lsintpt = self.latency_model.lut[i]
                f.write(f"{sstart},{send},{sslope},{ssintpt},{qstart},{qend},{qslope},{qsintpt},{lstart},{lend},{lslope},{lsintpt}\n")
        print(f"[VLITE] Latency regression model saved to {str(cpu_rgmodel_file)}")
        
        if self.latency_model_gpu:
            with open(gpu_rgmodel_file, 'w') as f:
                f.write("sstart,send,sslope,ssintpt,qstart,qend,qslope,qsintpt,lstart,lend,lslope,lsintpt\n")
                for i in range(len(self.latency_model.search)):
                    sstart, send, sslope, ssintpt = self.latency_model_gpu.search[i]
                    qstart, qend, qslope, qsintpt = self.latency_model_gpu.quantizer[i]
                    lstart, lend, lslope, lsintpt = self.latency_model_gpu.lut[i]
                    f.write(f"{sstart},{send},{sslope},{ssintpt},{qstart},{qend},{qslope},{qsintpt},{lstart},{lend},{lslope},{lsintpt}\n")
            print(f"[VLITE] GPU Latency regression model saved to {str(gpu_rgmodel_file)}")

    def save_latency_data(self, cfgs):
        nprobe = self.nprobe
        data_dir = Path(cfgs.result_dir).parent
        self.save_latency_model(cfgs)
        
        if self.latency_data is not None:
            data_path = data_dir / f"latency_breakdown_cpu_{nprobe}.csv"
            self.latency_data.to_csv(data_path, index=False)
            print(f"[VLITE] Latency data saved to {str(data_path)}")

        if self.latency_data_gpu is not None:
            data_path = data_dir / f"latency_breakdown_gpu_{nprobe}.csv"
            self.latency_data_gpu.to_csv(data_path, index=False)
            print(f"[VLITE] GPU Latency data saved to {str(data_path)}")
    
    # Staled method to process old CSV files
    def load_latency_data(self, cfgs):
        nprobe = self.nprobe
        results_dir = Path(cfgs.result_dir).parent
        
        latencyFileCPU = results_dir / f"latency_breakdown_cpu_{nprobe}.csv"
        latencyFileGPU = results_dir / f"latency_breakdown_gpu_{nprobe}.csv"
        
        rgmodelFileCPU = results_dir / f"latency_regression_cpu_{nprobe}.csv"
        rgmodelFileGPU = results_dir / f"latency_regression_gpu_{nprobe}.csv"

        if not latencyFileCPU.exists():
            return False

        self.latency_data = pd.read_csv(latencyFileCPU)
                
        if latencyFileGPU.exists():
            self.latency_data_gpu = pd.read_csv(latencyFileGPU)

        if rgmodelFileCPU.exists():
            self.latency_model = LatencyModel(cpu_num=self.nc, search=[], quantizer=[], lut=[])
            
            with open(rgmodelFileCPU, 'r') as f:
                lines = f.readlines()[1:]
                for line in lines:
                    parts = line.strip().split(',')
                    self.latency_model.add_data('search', float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
                    self.latency_model.add_data('quantizer', float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]))
                    self.latency_model.add_data('lut', float(parts[8]), float(parts[9]), float(parts[10]), float(parts[11]))
        
        if rgmodelFileGPU.exists():
            self.latency_model_gpu = LatencyModel(cpu_num=self.nc, search=[], quantizer=[], lut=[])
            with open(rgmodelFileGPU, 'r') as f:
                lines = f.readlines()[1:]
                for line in lines:
                    parts = line.strip().split(',')
                    self.latency_model_gpu.add_data('search', float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                    self.latency_model_gpu.add_data('quantizer', float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8]))
                    self.latency_model_gpu.add_data('lut', float(parts[9]), float(parts[10]), float(parts[11]), float(parts[12]))

        return True

    def collect_latency_data(self, cfgs):
        match = re.search(r'\b(H200|H100|A100|L40S)\b', torch.cuda.get_device_name(0))
        if not match:
            raise RuntimeError(f"Unknown GPU type: {torch.cuda.get_device_name(0)}")

        device_type = match.group(1)
        if device_type != cfgs.gpu_type:
            raise ValueError(f"Expected GPU={cfgs.gpu_type}, but running on {device_type}")
        
        print("[VLITE] Running batch-size sweep for latency profiling")
        asyncio.run(run_pipeline(cfgs))
        self.collect_latency_from_file(cfgs)

    def collect_latency_from_file(self, cfgs):
        cfg_cpy = deepcopy(cfgs)
        df = vLiteResults(cfg_cpy).read_raw_parquet()
        df= df.groupby('batch_size').mean().reset_index()
        if cfg_cpy.search_mode == 'all-gpu':
            self.latency_data_gpu = df
        else:
            self.latency_data = df
            
    def run_regression_model(self, cfgs):
        num_section = 4
        if self.latency_data.empty:
            raise ValueError("Latency data empty")
        
        def piecewise_linear_fit(x, y):
            x = x.to_numpy()
            y = y.to_numpy()
            qc_breaks = np.quantile(x, np.linspace(0, 1, num_section + 1))
            breaks = np.sort(np.unique(np.concatenate((qc_breaks, [1, 2]))))
            model = pwlf.PiecewiseLinFit(x, y)
            model.fit_with_breaks(breaks)
            return model

        data = self.latency_data[self.latency_data['batch_size'].between(1, self.nc)]
        s_model = piecewise_linear_fit(data['batch_size'], data['ann_search'])
        q_model = piecewise_linear_fit(data['batch_size'], data['quantize'])
        l_model = piecewise_linear_fit(data['batch_size'], data['lut_compute'] + data['lut_scan'])

        if self.latency_model is None:
            self.latency_model = LatencyModel(cpu_num=self.nc, search=[], quantizer=[], lut=[])
        
        for i in range(num_section):
            self.latency_model.add_data('search', s_model.fit_breaks[i], s_model.fit_breaks[i + 1], s_model.slopes[i], s_model.intercepts[i])
            self.latency_model.add_data('quantizer', q_model.fit_breaks[i], q_model.fit_breaks[i + 1], q_model.slopes[i], q_model.intercepts[i])
            self.latency_model.add_data('lut', l_model.fit_breaks[i], l_model.fit_breaks[i + 1], l_model.slopes[i], l_model.intercepts[i])
        
    def run_gpu_regression_model(self, cfgs):
        if self.latency_data_gpu is None or self.latency_data_gpu.empty:
            return
        
        for b in range(1, 65, self.nc):
            rng_start = b
            rng_end = min(b + self.nc - 1, 65)
            
            data = self.latency_data_gpu[self.latency_data_gpu['batch_size'].between(rng_start, rng_end)]
            
            if data.empty:
                continue
            
            search_slope, search_intpt = np.polyfit(data['batch_size'], data['ann_search'], 1)
            quantizer_slope, quantizer_intpt = np.polyfit(data['batch_size'], data['quantize'], 1)
            lut_total = data['lut_compute'] + data['lut_scan']
            lut_slope, lut_intpt = np.polyfit(data['batch_size'], lut_total, 1)

            if self.latency_model_gpu is None:
                self.latency_model_gpu = LatencyModel(cpu_num=self.nc, search=[], quantizer=[], lut=[])
            
            self.latency_model_gpu.add_data('search', rng_start, rng_end, search_slope, search_intpt)
            self.latency_model_gpu.add_data('quantizer', rng_start, rng_end, quantizer_slope, quantizer_intpt)
            self.latency_model_gpu.add_data('lut', rng_start, rng_end, lut_slope, lut_intpt)

    def estimate_latency(self, batch_size, min_hitrate=None):
        if self.latency_model is None:
            raise ValueError("Latency model missing")
        
        model = self.latency_model
        model_gpu = self.latency_model_gpu

        sslope, sintpt, qslope, qintpt, lslope, lintpt = 0, 0, 0, 0, 0, 0
        for i in range(len(model.search)):
            if model.search[i][0] <= batch_size <= model.search[i][1]:
                sslope, sintpt = model.search[i][2], model.search[i][3]
            if model.quantizer[i][0] <= batch_size <= model.quantizer[i][1]:
                qslope, qintpt = model.quantizer[i][2], model.quantizer[i][3]
            if model.lut[i][0] <= batch_size <= model.lut[i][1]:
                lslope, lintpt = model.lut[i][2], model.lut[i][3]
        
        cq_latency = qslope * batch_size + qintpt
        
        if min_hitrate is None:
            if self.hitrate_data is not None:
                if batch_size > len(self.hitrate_data):
                    raise ValueError(f"Batch size {batch_size} exceeds hitrate data length {len(self.hitrate_data)}.")
                missrate = 1.0 - self.hitrate_data[batch_size - 1]
            else:
                missrate = 1.0
        else:
            missrate = 1.0 - min_hitrate
        
        lut_latency = (lslope * batch_size + lintpt) * missrate
        search_latency_0 = sslope * batch_size + sintpt
        search_latency_1 = lut_latency + cq_latency

        if missrate == 0.0:
            if self.latency_model_gpu is not None:
                for i in range(len(model_gpu.search)):
                    if model_gpu.search[i][0] <= batch_size < model_gpu.search[i][1]:
                        sslope, sintpt = model_gpu.search[i][2], model_gpu.search[i][3]
                        search_latency_1 = sslope * batch_size + sintpt
                        break
            else:
                search_latency_1 /= 8
                
        return search_latency_0, search_latency_1, cq_latency, lut_latency

class HitRateEstimator:
    def __init__(self, nprobe, nq=0, max_var=0.25,
                 centroid_ids=None, centroid_cdf=None):
        self.nq = nq
        self.nprobe = nprobe
        self.max_var = max_var
        self.cids = centroid_ids
        self.ccdf = centroid_cdf
    
    def save_centroids_data(self, cfgs):
        results_dir = Path(cfgs.database_dir) / cfgs.index / 'metadata'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        ids_path = results_dir / f"ordered_centroids_{cfgs.search_nprobe}.npz"
        cdf_path = results_dir / f"centroid_cdf_{cfgs.search_nprobe}.npz"
        meta_path = results_dir / f"centroid_meta_{cfgs.search_nprobe}.txt"
        
        np.savez(ids_path, ids=self.cids)
        np.savez(cdf_path, cdf=self.ccdf)
        with open(meta_path, 'w') as f:
            f.write(f"nq={self.nq},nprobe={self.nprobe},max_var={self.max_var}\n")

    def load_centroids_data(self, cfgs):
        results_dir = Path(cfgs.database_dir) / cfgs.index / 'metadata'
        
        ids_path = results_dir / f"ordered_centroids_{cfgs.search_nprobe}.npz"
        cdf_path = results_dir / f"centroid_cdf_{cfgs.search_nprobe}.npz"
        meta_path = results_dir / f"centroid_meta_{cfgs.search_nprobe}.txt"
        
        if not ids_path.exists() or not cdf_path.exists():
            return False
        
        data = np.load(ids_path)
        self.cids = data['ids']
        
        data = np.load(cdf_path)
        self.ccdf = data['cdf']
        
        with open(meta_path, 'r') as f:
            meta = f.readline().strip().split(',')
            self.nq = int(meta[0].split('=')[1])
            self.nprobe = int(meta[1].split('=')[1])
            self.max_var = float(meta[2].split('=')[1])

        return True

    def collect_centroid_data(self, cfgs):
        nprobe = self.nprobe
        queries, _ = prepare_queries(cfgs, qtype="train", need_texts=False)
        index = ShardedIndex(cfgs, nprobe)
        index.init_index(verbose=True)
        
        index_metadata_writer(cfgs, index)
        
        if len(queries) > 10000:
            queries = queries[:10000]
        self.nq = len(queries)
        
        quantizer = faiss.downcast_index(index.index.quantizer)
        
        if isinstance(quantizer, faiss.IndexHNSWFlat):
            quantizer.hnsw.efSearch = int(nprobe * 1.5)
        
        Dq, Iq = quantizer.search(queries, nprobe)
        
        centroid_freq = np.bincount(Iq.ravel(), minlength=quantizer.ntotal)
        ordered_ids = np.argsort(centroid_freq)[::-1]
        sorted_freq = centroid_freq[ordered_ids]
        cdf_freq = np.cumsum(sorted_freq) / (nprobe * len(queries))
        
        # -- get variance of hit rate at 0.5
        half_point = np.argmax(cdf_freq >= 0.5)
        cached_centroids = ordered_ids[:half_point]
        
        mask = np.isin(Iq, list(cached_centroids))
        hit_rates = mask.sum(axis=1).astype(np.float32) / nprobe
        
        var = np.var(hit_rates)
        mean = np.mean(hit_rates)
        print(f"[VLITE] Centroid Analytics: Mean Hit Rate: {mean:.4f}, Variance: {var:.4f} @ 0.5 CDF")

        self.max_var = var
        self.cids = ordered_ids
        self.ccdf = cdf_freq

    def estimate_var(self, exp_mean):
        return 4 * self.max_var * exp_mean * (1 - exp_mean)
    
    def estimate_alpha_beta(self, exp_mean):
        var = self.estimate_var(exp_mean)
        common = exp_mean * (1 - exp_mean) / var - 1
        alpha = exp_mean * common
        beta = (1 - exp_mean) * common
        return alpha, beta
    
    def compute_min_hitrate(self, batch_size, exp_mean):
        if exp_mean == 1.0:
            return 1.0
        elif exp_mean == 0.0:
            return 0.0
        
        a, b = self.estimate_alpha_beta(exp_mean)
        from scipy.stats import beta
        from scipy.integrate import quad
        
        def integrand(x):
            f = beta.pdf(x, a, b)
            F = beta.cdf(x, a, b)
            return x * batch_size * ((1 - F) ** (batch_size - 1)) * f
        
        integral, _ = quad(integrand, 0, 1)
        return integral

def hitrate_binarysearch(exp_min_hitrate, batch_size, hitrate_estimator):
    if exp_min_hitrate <= 0.0 or exp_min_hitrate >= 1.0:
        return exp_min_hitrate
    
    low = 0.0
    high = 1.0
    tol = 1e-3
    exp_mean_hitrate = None
    
    while high - low > tol:
        mid = (low + high) / 2
        min_hitrate_val = hitrate_estimator.compute_min_hitrate(batch_size, mid)
        if min_hitrate_val < exp_min_hitrate:
            low = mid
        else:
            high = mid
        exp_mean_hitrate = mid
            
    if exp_mean_hitrate is None:
        raise ValueError("Failed to estimate expected mean hitrate.")
    
    return exp_mean_hitrate

def search_exp_mean_hitrate_ceil(optimal_batch_size, latency_estimator, hitrate_estimator, serving_time_target):
    from math import ceil
    prev_batch_size = ceil(optimal_batch_size)
    best_expected_mean_hitrate = 1.0
    best_expected_min_hitrate = 1.0
    best_batch_size = prev_batch_size

    while True:
        _, search_time, cq_time, lut_time = latency_estimator.estimate_latency(prev_batch_size)
        expected_min_hitrate = np.clip(
            np.float64((search_time - serving_time_target) / lut_time), 0.0, 1.0)
        expected_mean_hitrate = hitrate_binarysearch(expected_min_hitrate, prev_batch_size, hitrate_estimator)
        if expected_mean_hitrate >= best_expected_mean_hitrate:
            break
        best_expected_mean_hitrate = expected_mean_hitrate
        best_expected_min_hitrate = expected_min_hitrate
        best_batch_size = prev_batch_size
        prev_batch_size += 1
        
    return best_batch_size, best_expected_mean_hitrate, best_expected_min_hitrate
    
def search_exp_mean_hitrate_floor(optimal_batch_size, latency_estimator, hitrate_estimator, throughput_target):
    from math import floor
    prev_batch_size = floor(optimal_batch_size)
    best_expected_mean_hitrate = 1.0
    best_expected_min_hitrate = 1.0
    best_batch_size = prev_batch_size
    
    if prev_batch_size < 1:
        return 1, 1.0, 1.0
    
    while True:
        _, search_time, cq_time, lut_time = latency_estimator.estimate_latency(prev_batch_size)
        expected_min_hitrate = np.clip(
            np.float64((search_time - prev_batch_size/throughput_target) / lut_time), 0.0, 1.0)
        expected_mean_hitrate = hitrate_binarysearch(expected_min_hitrate, prev_batch_size, hitrate_estimator)
        if expected_mean_hitrate >= best_expected_mean_hitrate:
            break
        best_expected_mean_hitrate = expected_mean_hitrate
        best_expected_min_hitrate = expected_min_hitrate
        best_batch_size = prev_batch_size
        prev_batch_size -= 1
        if prev_batch_size < 1:
            break

    return best_batch_size, best_expected_mean_hitrate, best_expected_min_hitrate 

def partition_point_iteration(cfgs, init_throughput, throughput_target, latency_estimator, hitrate_estimator):
    latency_target = float(cfgs.search_slo / 1000)
    
    latency_margin = 1
    serving_time_target = latency_target / (1 + latency_margin)
    optimal_batch_size = throughput_target * serving_time_target

    batch_size_1, exp_mean_hitrate_1, exp_min_hitrate_1 = search_exp_mean_hitrate_floor(
        optimal_batch_size, latency_estimator, hitrate_estimator, throughput_target)

    batch_size_2, exp_mean_hitrate_2, exp_min_hitrate_2 = search_exp_mean_hitrate_ceil(
        optimal_batch_size, latency_estimator, hitrate_estimator, serving_time_target)

    if exp_mean_hitrate_1 <= exp_mean_hitrate_2:
        optimal_batch_size = batch_size_1
        if init_throughput < 25:
            exp_mean_hitrate =  (exp_mean_hitrate_1 * 3 + exp_mean_hitrate_2) / 4
        else:
            exp_mean_hitrate = exp_mean_hitrate_1
    else:
        optimal_batch_size = batch_size_2
        if init_throughput < 25:
            exp_mean_hitrate = (exp_mean_hitrate_2 * 3 + exp_mean_hitrate_1) / 4
        else:
            exp_mean_hitrate = exp_mean_hitrate_2

    # Finally get paritioning point
    if 0.0 < exp_mean_hitrate < 1.0:
        low = 0.0
        nlist = hitrate_estimator.cids.size
        high = exp_mean_hitrate
        scaler = nlist * hitrate_estimator.nprobe  # partitioning resolution
        partitioning_point = None

        while int(scaler * (high - low)) > 1:
            mid = (low + high) / 2
            idx = min(int(mid * nlist), nlist - 1)
            ccdf_val = hitrate_estimator.ccdf[idx]

            if ccdf_val < exp_mean_hitrate:
                low = mid
            else:
                high = mid
                partitioning_point = mid
            
        if partitioning_point is None:
            raise ValueError("Failed to find partitioning point.")

        num_pt_centroids = int(partitioning_point * nlist)
    elif exp_mean_hitrate == 1.0:
        partitioning_point = 1.0
        num_pt_centroids = hitrate_estimator.cids.size
    else:
        partitioning_point = 0.0
        num_pt_centroids = 0
    
    return hitrate_estimator.cids[:num_pt_centroids], partitioning_point, optimal_batch_size, exp_mean_hitrate

def partitioning_point_search(cfgs, latency_estimator, hitrate_estimator):
    _, list_sizes = index_metadata_reader(cfgs)

    gpu_mem = load_gpu()[cfgs.gpu_type]
    model_mem = cfgs.model_cfg['memory_gb']
    kv_mem0 = gpu_mem - model_mem

    throughput0 = cfgs.get_tput_ceiling()

    ppt_low = 0.0
    ppt_high = 0.99
    best_ppt = None
    best_exp_batch_size = None
    best_exp_mean_hitrate = None
    best_throughput = 0.0
    partitioned_cids = None
    mem_calc = IndexMemoryCalculator(cfgs, list_sizes)

    while ppt_high - ppt_low > 1e-4:
        mid = (ppt_low + ppt_high) / 2
        index_mem = mem_calc.get_total_size(mid, partitioned=True)
        kv_mem = kv_mem0 - index_mem

        if kv_mem <= 0:
            ppt_high = mid
            continue

        throughput_ = throughput0 * (kv_mem / kv_mem0)

        result = partition_point_iteration(cfgs, throughput0, throughput_, latency_estimator, hitrate_estimator)
        if result is None:
            ppt_high = mid
            continue

        partitioned_cids_, partitioning_point_, exp_batch_size, exp_mean_hitrate = result
        if partitioning_point_ is None:
            ppt_high = mid
            continue

        # Binary search update
        if partitioning_point_ > mid:
            ppt_low = partitioning_point_
        else:
            ppt_high = mid

        # Save best result
        if best_ppt is None or partitioning_point_ > best_ppt:
            best_ppt = partitioning_point_
            best_exp_batch_size = exp_batch_size
            best_exp_mean_hitrate = exp_mean_hitrate
            best_throughput = throughput0 * (kv_mem0 - mem_calc.get_total_size(partitioning_point_, partitioned=True)) / kv_mem0
            partitioned_cids = partitioned_cids_

    return best_ppt, best_exp_batch_size, best_exp_mean_hitrate, best_throughput, partitioned_cids

def save_partitioned_centroids(cfgs, partitioned_cids, partitioning_point, exp_batch_size, exp_mean_hitrate, md, list_sizes):
    nprobe = cfgs.search_nprobe
    centroid_dir = Path(cfgs.database_dir) / cfgs.index / cfgs.model / f"{cfgs.num_gpus}gpus" / 'shards'
    
    if not centroid_dir.exists():
        centroid_dir.mkdir(parents=True, exist_ok=True)
    
    ids_path = centroid_dir / f"{cfgs.search_slo}ms_cids_{nprobe}.npz"
    mtd_path = centroid_dir / f"{cfgs.search_slo}ms_meta_{nprobe}.txt"
    tput = cfgs.get_tput_ceiling()

    for line in md:
        if "ntotal" in line:
            ntotal = int(line.split(":", 1)[1].strip())
        elif "nlist" in line:
            nlist = int(line.split(":", 1)[1].strip())

    num_pt_vectors = 0
    for i, cid in enumerate(partitioned_cids):
        num_pt_vectors += list_sizes[cid]
    
    plist = len(partitioned_cids) / nlist
    ptotal = num_pt_vectors / ntotal
    
    with open(mtd_path, 'w') as f:
        f.write(f"Model                          : {cfgs.model}\n")
        f.write(f"Target Latency                 : {cfgs.search_slo}ms\n")
        f.write(f"Target Throughput              : {tput}\n")
        f.write(f"Partitioning point             : {partitioning_point:.4f}\n")
        f.write(f"Number of Partitioned centroids: {len(partitioned_cids)} ({plist * 100:.2f}%) of {nlist}\n")
        f.write(f"Number of Partitioned vectors  : {num_pt_vectors} ({ptotal * 100:.2f}%) of {ntotal}\n")
        f.write(f"Expected batch size            : {exp_batch_size:.2f}\n")
        f.write(f"Expected mean hitrate          : {exp_mean_hitrate:.4f}\n")
    
    print(f"[VLITE]   - Number of partitioned centroids: {len(partitioned_cids)} ({plist * 100:.2f}%) of {nlist}")
    print(f"[VLITE]   - Number of partitioned vectors: {num_pt_vectors} ({ptotal * 100:.2f}%) of {ntotal}")
    print(f"[VLITE] Partitioned centroids metadata saved to {str(mtd_path)}")
    
    np.savez(ids_path, ids=partitioned_cids)
    print(f"[VLITE] Partitioned centroids saved to {str(ids_path)}")

def index_metadata_writer(cfgs, index=None):
    index_dir = Path(cfgs.database_dir) / cfgs.index / 'metadata'
    index_dir.mkdir(parents=True, exist_ok=True)

    lsizesFile = Path(index_dir) / "list_sizes.npz"
    metaFile = Path(index_dir) / "index_meta.txt"
    
    if lsizesFile.exists() and metaFile.exists():
        return
    
    if index is None:
        index = ShardedIndex(cfgs)
        index.init_index(verbose=True)
        
    index_core = faiss.downcast_index(index.index)
    
    nlist = index_core.invlists.nlist
    ntotal = index_core.ntotal
    
    list_sizes = np.array([index_core.invlists.list_size(i) for i in range(nlist)], dtype=np.int64)
    
    avg_list_size = np.mean(list_sizes)
    max_list_size = np.max(list_sizes)
    min_list_size = np.min(list_sizes)
    
    np.savez(lsizesFile, sizes=list_sizes)
    with open(metaFile, 'w') as f:
        f.write(f"Number of Lists (nlist)        : {nlist}\n")
        f.write(f"Number of Vectors (ntotal).    : {ntotal}\n")
        f.write(f"Average List Size              : {avg_list_size:.2f}\n")
        f.write(f"Maximum List Size              : {max_list_size}\n")
        f.write(f"Minimum List Size              : {min_list_size}\n")
     
    # update index.json   
    index_json = load_index()
    index_json[cfgs.index]['ntotal'] = ntotal
    index_json[cfgs.index]['nlist'] = nlist
    write_json('index.json', index_json)

def index_metadata_reader(cfgs):
    index_dir = Path(cfgs.database_dir) / cfgs.index / 'metadata'
    lsizesFile = Path(index_dir) / "list_sizes.npz"
    metaFile = Path(index_dir) / "index_meta.txt"
    
    if not lsizesFile.exists() or not metaFile.exists():
        print(f"[VLITE] Index metadata files not found in {index_dir}.")
        return
    
    data = np.load(lsizesFile)
    list_sizes = data['sizes']
    
    with open(metaFile, 'r') as f:
        meta_data = f.readlines()
    
    return meta_data, list_sizes

def assess_disaggregated_solutions(cfgs, latency_estimator, hitrate_estimator):
    if cfgs.search_nprobe > 4096:
        return []
    
    from math import ceil
    mem_calc = IndexMemoryCalculator(cfgs)
    index_mem = mem_calc.get_total_size(1.0, partitioned=False)
    gpu_mem = load_gpu()[cfgs.gpu_type]
    throughput0 = cfgs.get_tput_ceiling()
    tp_size = cfgs.tp_size
    
    latency_target = float(cfgs.search_slo / 2000)
    n1 = tp_size
    n2 = ceil(index_mem / (gpu_mem * tp_size)) if gpu_mem < index_mem else tp_size
    max_n = cfgs.num_gpus - tp_size
    
    sol = []
    for n in range(n1, min(n2, max_n)+1, tp_size):
        ppt, _, mhr = find_partition_point_ngpu(cfgs, n, hitrate_estimator)
        throughput = throughput0 * (cfgs.num_gpus - n) / cfgs.num_gpus
        batch_size = int(throughput * latency_target) + 1
        
        while True:
            min_hr = hitrate_estimator.compute_min_hitrate(batch_size, mhr)
            search_time = latency_estimator.estimate_latency(batch_size, min_hr)[1]
            if search_time <= latency_target:
                break
            elif batch_size == 1:
                break
            else:
                batch_size -= 1
                throughput = batch_size / latency_target
        
        if search_time <= latency_target:
            sol.append((n, throughput, batch_size, search_time))
    return sol

def find_partition_point_ngpu(cfgs, n, hitrate_estimator):
    gpu_mem = load_gpu()[cfgs.gpu_type]
    nlist = load_index()[cfgs.index]
    _, list_sizes = index_metadata_reader(cfgs)
    mem_calc = IndexMemoryCalculator(cfgs, list_sizes)
    index_mem = mem_calc.get_total_size(1.0, partitioned=False)
    
    if index_mem <= gpu_mem * n:
        return 1.0, hitrate_estimator.cids, 1.0
    else:
        high = 1.0
        low = 0.0
        
        while high - low > 1e-3:
            mid = (low + high) / 2
            ppt_mem = mem_calc.get_total_size(mid, partitioned=False)
            if ppt_mem < n * gpu_mem:
                low = mid
            else:
                high = mid
        
        ppt = high
        p_cid = int(ppt * hitrate_estimator.cids.size)
        partitioned_centroids = hitrate_estimator.cids[:p_cid]
        expected_mean_hitrate = hitrate_estimator.ccdf[p_cid - 1] if p_cid > 0 else 0.0

        return ppt, partitioned_centroids, expected_mean_hitrate

def plot_cdf(np_cdf, nprobe):
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, len(np_cdf))

    plt.figure(figsize=(8, 6))
    plt.plot(x, np_cdf, linestyle='-', color='blue')
    plt.title('CDF of Latency')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Cumulative Probability')
    plt.grid()
    
    plt.savefig(f"cdf_nprobe_{nprobe}.pdf")

def profile(cfgs):
    nprobe = cfgs.search_nprobe
    num_cpu = cfgs.num_gpus * (4 if cfgs.gpu_type == "L40S" else 8)
    print("[VLITE] Profiling latencies...")
    
    lestimator = LatencyEstimator(num_cpu, nprobe)
    
    if lestimator.load_latency_data(cfgs):
        lestimator.collect_latency_from_file(cfgs)
    else:
        lestimator.collect_latency_data(cfgs)

    lestimator.run_regression_model(cfgs)
    lestimator.run_gpu_regression_model(cfgs)
    lestimator.save_latency_data(cfgs)
    
    print("[VLITE] Profiling hitrates...")
    hestimator = HitRateEstimator(nprobe)
    if not hestimator.load_centroids_data(cfgs):
        hestimator.collect_centroid_data(cfgs)
        plot_cdf(hestimator.ccdf, nprobe)  
    hestimator.save_centroids_data(cfgs)    
    
    index_metadata_writer(cfgs)
    md, list_sizes = index_metadata_reader(cfgs)
    
    if cfgs.search_slo <= 0 or cfgs.sweep:
        cfgs.search_slo = load_index()[cfgs.index]['slo']
    
    t0 = time.time()
    
    cfgs.set_result_paths()
    dsol = assess_disaggregated_solutions(cfgs, lestimator, hestimator)

    t1 = time.time()

    best_ppt, best_batch_size, best_exp_mean_hitrate, best_throughput, partitioned_cids = partitioning_point_search(
        cfgs, lestimator, hestimator) 
    best_min_hitrate = hestimator.compute_min_hitrate(best_batch_size, best_exp_mean_hitrate)
    best_time = lestimator.estimate_latency(best_batch_size, best_min_hitrate)[1]
    index_mem = IndexMemoryCalculator(cfgs, list_sizes).get_total_size(best_ppt, partitioned=True)
    
    t3 = time.time()

    print("===" * 40)
    print(f"[VLITE] Disaggregated Solution for {cfgs.search_slo}ms")
    for sol in dsol:
        n, tput, bsize, search_time = sol
        print(f"[VLITE] ** {n} GPUs: Throughput: {tput:.2f} rps, Batch Size: {bsize}, Search Time: {search_time:.4f} seconds")
    print("---" * 40)
    print(f"[VLITE] Colocated Solution for {cfgs.search_slo}ms")
    print(f"[VLITE] ** Partitioning Point: {best_ppt:.4f}, Expected Batch Size: {best_batch_size:.2f}, ")
    print(f"[VLITE] ** Expected Mean Hit Rate: {best_exp_mean_hitrate:.4f}, Estimated Throughput: {best_throughput:.2f} rps")
    print(f"[VLITE] ** Estimated Search Time: {best_time:.4f} seconds")
    print(f"[VLITE] ** Estimated Per GPU Index Memory: {index_mem:.2f} GiB")
    print(f"---" * 40)

    scheme = ""
    best_n = -1
    for n, tput, bsize, search_time in dsol:
        if best_throughput <= tput:
            best_throughput = tput
            scheme = f"Disaggregated, {n}"
            best_batch_size = bsize
            best_time = search_time
            best_n = n
            best_ppt, partitioned_cids, best_exp_mean_hitrate = find_partition_point_ngpu(cfgs, n, hestimator)

    if scheme == "":
        scheme = "Colocated"
    
    print(f"[VLITE] Best Partitioning Scheme: {scheme}")
    if scheme == "Colocated":
        print(f"[VLITE]   - Partitioning Point: {best_ppt:.4f}")
    else:
        print(f"[VLITE]   - Partitioning Point: {best_ppt:.4f} (Disaggregated with {best_n} GPUs)")
    print(f"[VLITE]   - Expected Batch Size: {best_batch_size:.2f}")
    print(f"[VLITE]   - Expected Mean Hit Rate: {best_exp_mean_hitrate:.4f}")
    print(f"[VLITE]   - Estimated Throughput: {best_throughput:.2f} rps")
    print(f"[VLITE]   - Estimated Search Time: {best_time * 1000:.2f} ms")

    t4 = time.time()

    if scheme == "Colocated":
        save_partitioned_centroids(
            cfgs, partitioned_cids, best_ppt, best_batch_size, best_exp_mean_hitrate, md, list_sizes)
        
        t5 = time.time()

    print(f"[VLITE] Time taken for latency profiling: {t1 - t0:.2f} seconds")
    print(f"[VLITE] Time taken for partitioning point search: {t3 - t1:.2f} seconds")
    if scheme == "Colocated":
        print(f"[VLITE] Time taken for saving partitioned centroids: {t5 - t4:.2f} seconds")
