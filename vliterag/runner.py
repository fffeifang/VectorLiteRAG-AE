import os
import gc
import time
import queue
import psutil
import asyncio
import threading

import torch
import torch.distributed as dist

from vliterag.args import parse_args
from vliterag.configs import vLiteConfigs
from vliterag.results import vLiteResults
from vliterag.utils import get_gpu_memory, prepare_queries
from vliterag.engines import vLiteQueues, RequestGenerator, LLMEngine, ANNSEngine, DocRetriever, model_cache

from transformers import AutoTokenizer
from multiprocessing import Process, set_start_method

set_start_method('spawn', force=True)

def faiss_wrapper(cfg, reqGenDoneFlag, annsInitFlag, stopFlag,
                  searchQueue, retrieverQueue, llmQueues, statQueue,
                  query_vectors, tpotValue):
    anns_engine = ANNSEngine(cfg, reqGenDoneFlag, annsInitFlag, stopFlag, searchQueue, 
                             retrieverQueue, llmQueues, statQueue, query_vectors, tpotValue)
    anns_engine.init_engine()
    anns_engine.run_engine()

def vllm_wrapper(cfg, rank, llmQueue, outputQueue, 
                llmInitFlag, reqGenDoneFlag, stopFlag, query_texts=None, tpotValue=None):

    engine = LLMEngine(cfg, rank, llmQueue, outputQueue, llmInitFlag, 
                       reqGenDoneFlag, stopFlag, query_texts, tpotValue)
    asyncio.run(engine.run_engine())

def init_engines(cfg, queues: vLiteQueues): # tuple[list[Process], faiss.index or index class]:
    queues.stopFlag.clear()
    num_workers = cfg.llm_workers

    ts0 = time.time()
    query_vectors, query_texts = prepare_queries(cfg, need_texts=False)
    
    time.sleep(10)
    annsProcess = Process(
        target=faiss_wrapper,
        args=(
            cfg,
            queues.reqGenDoneFlag,
            queues.annsInitFlag,
            queues.stopFlag,
            queues.searchQueue,
            queues.retrieverQueue,
            queues.llmQueue,
            queues.statQueue,
            query_vectors,
            queues.tpotValue
        )
    )
    annsProcess.start()
    queues.annsInitFlag.wait()
    ts1 = time.time()
    print(f"[VLITE] ANN Engine initialized in {ts1 - ts0:.1f} seconds.")
    
    gpu_mem, faiss_use, free_mem = get_gpu_memory()
    if cfg.search_mode in ['cpu', 'ded-gpu']:
        gpu_util = cfg.gpu_util
    else:
        gpu_util = free_mem / gpu_mem - 0.05

    print(f"[VLITE] GPU Memory: {gpu_mem/1024:.2f} GiB, FAISS Usage: {faiss_use/1024:.2f} GiB, " + \
        f"Free Memory: {free_mem/1024:.2f} GiB, GPU Utilization for vLLM: {gpu_util:.2f}")

    cfg.gpu_util = gpu_util
    cfg.set_result_paths()
    llmProcesses = []
    
    for rank in range(num_workers):
        p = Process(
            target=vllm_wrapper,
            args=(
                cfg,
                rank,
                queues.llmQueue[rank],
                queues.outQueue[rank],
                queues.llmInitFlags[rank],
                queues.reqGenDoneFlag,
                queues.stopFlag,
                query_texts,
                queues.tpotValue if rank == 0 else None  # Only the first worker tracks LLM TPS
            )
        )
        p.start()
        llmProcesses.append(p)

    return annsProcess, llmProcesses

def close_engines(annsProcess, llmProcesses, queues: vLiteQueues):
    queues.set_flags("stop", True)
    queues.set_flags("initl", False)
    
    for p in llmProcesses:
        p.join(timeout=5)
        if p.is_alive():
            print(f"[VLITE] Process {p.pid} is still alive, terminating...")
            p.terminate()
            p.join()
            
    if annsProcess.is_alive():
        annsProcess.join(timeout=5)
        if annsProcess.is_alive():
            print(f"[VLITE] ANN process {annsProcess.pid} is still alive, terminating...")
            annsProcess.terminate()
            annsProcess.join()
            
    if dist.is_initialized():
        dist.barrier()  # Ensure all processes reach this point before destroying the group
        dist.destroy_process_group()
        
    gc.collect()
    torch.cuda.empty_cache()

def collect_anns_results(cfg, queues: vLiteQueues):
    anns_result = dict()
    if cfg.search_mode:        
        while not len(anns_result) == cfg.total_requests:
            try:
                stats = queues.statQueue.get(timeout=1)
                if stats[0] < cfg.warmup_requests or stats[0] >= cfg.total_requests + cfg.warmup_requests:
                    continue
                anns_result[stats[0] - cfg.warmup_requests] = stats[1:]
            except queue.Empty:
                continue
            time.sleep(0.001)
            
    return dict(sorted(anns_result.items()))

def collect_llm_results(cfg, queues: vLiteQueues):
    llm_results = dict()
    while not len(llm_results) == cfg.total_requests:
        for q in queues.outQueue:
            try:
                result = q.get(timeout=1)
                if result[0] < cfg.warmup_requests or result[0] >= cfg.total_requests + cfg.warmup_requests:
                    continue
                llm_results[result[0] - cfg.warmup_requests] = result[1:]
            except queue.Empty:
                continue
            time.sleep(0.001)
    return dict(sorted(llm_results.items()))

def calc_metrics(anns_results, llm_results, vlite_results, tokenizer):    
    for rid, anns_result in anns_results.items():
        llm_result = llm_results.get(rid, None)
    
        if anns_result is None or llm_result is None:
            raise ValueError(f"Results for request {rid} are missing.")
        
        bid, batch_size, tIdling, tsArrival, tsFinalArrival, tsStart, tBreakdown, hitrate = anns_result
        prompt, tsCallback, tsSubmit, tsPrefill, tsCompletion = llm_result

        input_len = len(tokenizer.encode(prompt))
        idling = tIdling
        batching = tsFinalArrival - tsArrival
        pending = tsStart - tsFinalArrival
        queuing1 = tsStart - tsArrival
        searching = tsCallback - tsStart
        
        queuing2 = tsSubmit - tsCallback
        prefilling = tsPrefill - tsSubmit
        generating = tsCompletion - tsPrefill
        ttft = queuing1 + searching + queuing2 + prefilling
        end2end = tsCompletion - tsArrival
        
        vlite_results.add_anns_result(rid, bid, batch_size,
                                    queuing1, batching, pending, idling, searching, 
                                    ttft, tBreakdown, hitrate)
        vlite_results.add_llm_result(rid, queuing2, prefilling, generating, end2end)
        
        if rid == 0:
            vlite_results.tot_time = tsArrival
        
        if rid == len(anns_results) - 1:
            vlite_results.tot_time = tsCompletion - vlite_results.tot_time
            
    vlite_results.compute_averages()
    vlite_results.print_results()

def collect_results(cfg, results, queues):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_cfg['name'],
        cache_dir=model_cache,
        use_fast=True)

    print(f"[VLITE] Collecting Output Data...")
    anns_results = collect_anns_results(cfg, queues)
    print(f"[VLITE] Collected {len(anns_results)} ANN results.")
    llm_results = collect_llm_results(cfg, queues)
    print(f"[VLITE] Collected {len(llm_results)} LLM results.")
    calc_metrics(anns_results, llm_results, results, tokenizer)
    print(f"[VLITE] Collected all results.")

    results.save_results()
    queues.clear_queues()

async def run_pipeline(cfg):
    queues = vLiteQueues(cfg)
    results = vLiteResults(cfg)
    annsProcess, llmProcesses = init_engines(
        cfg, queues)
    
    while True:
        dr = DocRetriever(cfg, queues)
        rg = RequestGenerator(cfg, queues, poisson=True)
        req_generator = threading.Thread(target=rg.generate, daemon=True)
        doc_retriever = threading.Thread(target=dr.run, daemon=True)
        
        doc_retriever.start()
        req_generator.start()
        req_generator.join()
        doc_retriever.join()
        
        collect_results(cfg, results, queues)
        
        if cfg.update_and_sweep():
            results = vLiteResults(cfg)
        else:
            break

    print("[VLITE] Exiting...")
    close_engines(annsProcess, llmProcesses, queues)