import os
import uuid
import time
import queue
import plyvel
import asyncio
import numpy as np

from vliterag.configs import vLiteConfigs
from vliterag.utils import random_prompt
from multiprocessing import Queue, Event, Value
from index.index_wrapper import BaseGPUIndex, PartitionedIndex, ShardedIndex
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

model_cache = '/tmp/models'

class vLiteQueues:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        nranks = cfgs.llm_workers
        
        # Queues for different stages of the pipeline
        self.searchQueue = Queue()
        self.retrieverQueue = Queue()
        self.statQueue = Queue()
        self.llmQueue = [Queue() for _ in range(nranks)]
        self.outQueue = [Queue() for _ in range(nranks)]

        # Flags to indicate completion of various stages
        self.annsInitFlag = Event()
        self.llmInitFlags = [Event() for _ in range(nranks)]
        self.reqGenDoneFlag = Event()
        self.stopFlag = Event()
        self._flag_map = {
            'stop': 'stopFlag',
            'inita': 'annsInitFlag',
            'initl': 'llmInitFlags',
            'request': 'reqGenDoneFlag',
        }
        
        # Values for tracking performance metrics
        self.tpotValue = Value('f', 0.0)  # Moving average of LLM Tokens Per Second
        
    def clear_queues(self):
        def _drain_queue(q):
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
        _drain_queue(self.searchQueue)
        _drain_queue(self.retrieverQueue)
        _drain_queue(self.statQueue)
        for q in self.llmQueue:
            _drain_queue(q)
        for q in self.outQueue:
            _drain_queue(q)
            
        self.tpotValue.value = 0.0

    def set_flags(self, flag_name, value: bool):
        flag_attr = self._flag_map.get(flag_name)
        if flag_attr is None or not hasattr(self, flag_attr):
            raise AttributeError(f"Invalid flag name: {flag_name}")
        flag = getattr(self, flag_attr)
        if isinstance(flag, list):
            for f in flag:
                f.set() if value else f.clear()
        else:
            flag.set() if value else flag.clear()

class RequestGenerator:
    def __init__(self, cfgs, queues, poisson=True):
        self.cfgs = cfgs
        self.poisson = poisson
        self.search_queue = queues.searchQueue
        self.llm_queue = queues.llmQueue
        self.annsInitFlag = queues.annsInitFlag
        self.llmInitFlags = queues.llmInitFlags
        self.doneFlag = queues.reqGenDoneFlag
    
    def generate(self):
        poisson = self.poisson
        num_workers = self.cfgs.llm_workers
        arrival_rate = self.cfgs.arrival_rate
        wu_requests = self.cfgs.warmup_requests
        num_requests = self.cfgs.total_requests + wu_requests * 2
        
        self._wait_for_init()
        self.doneFlag.clear()
        
        prompt_pool = random_prompt(self.cfgs.input_len, num_requests)
        
        print(f"[VLITE] Starting request generation with {num_requests} requests, ")
        rid = 0
        while rid < num_requests:
            if arrival_rate > 0:
                if wu_requests <= rid < self.cfgs.total_requests + wu_requests:
                    sleep_time = 1.0 / arrival_rate if not poisson else np.random.exponential(1.0 / arrival_rate)
                else:
                    sleep_time = 1.0
                time.sleep(sleep_time)
            if self.cfgs.search_mode:   # RAG
                self.search_queue.put((rid, time.time(), prompt_pool[rid]))
            else:                           # Standalone LLM
                self.llm_queue[rid % num_workers].put((rid, time.time(), prompt_pool[rid]))
            rid += 1

        print(f"[VLITE] Request generation completed. Total requests: {rid}.")
        self.doneFlag.set()
        
    def _wait_for_init(self):
        while not all(f.is_set() for f in self.llmInitFlags): 
            time.sleep(1)

class LLMEngine:
    def __init__(self, cfgs, rank, 
                llmQueue, outputQueue, initDoneFlag, reqGenDoneFlag, stopFlag,
                query_texts=None, tpotValue=None):

        self.cfgs = cfgs
        self.rank = rank
        self.engine = None
        self.query_texts = query_texts
        self.inQueue = llmQueue
        self.outQueue = outputQueue
        self.init_done = initDoneFlag
        self.req_gen_done = reqGenDoneFlag
        self.stop_flag = stopFlag
        self.tokenizer = None
        
        self.tpot_tracker = []
        self.tpotValue = tpotValue

    def init_engine(self):
        rank = self.rank
        tp_size = self.cfgs.tp_size
        
        gpu_list = ",".join(str(i) for i in range(rank * tp_size, (rank + 1) * tp_size))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

        engine_args = AsyncEngineArgs(
            model=self.cfgs.model_cfg['name'],
            download_dir=model_cache,
            gpu_memory_utilization=self.cfgs.gpu_util,
            tensor_parallel_size=self.cfgs.tp_size,
            enforce_eager=self.cfgs.eager_mode,
            disable_log_requests=False,
            enable_prefix_caching=False,
            enable_expert_parallel=False,
            max_model_len=5120,
            max_num_batched_tokens=16384
            )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        pid = os.getpid()
        print(f"[VLITE] LLM Engine initialized with PID {pid}, rank {rank}, and TP size {tp_size}.")

    def log_tpot(self, value):
        if self.tpotValue is not None:
            self.tpot_tracker.append(value / self.cfgs.output_len)
            self.tpot_tracker = self.tpot_tracker[-10:]  # Keep last 10 values
            self.tpotValue.value = np.mean(self.tpot_tracker) if len(self.tpot_tracker) > 0 else 0.0

    async def streaming(self, prompt, rid, sampling_params, ts_callback):
        ts_submit = time.time()
        prefill = True
 
        responseGenerator = self.engine.generate(
            prompt,
            sampling_params,
            request_id=str(uuid.uuid4().hex[:16]) + f"-{rid}"
        )
        
        async for output in responseGenerator:
            if output.outputs:
                ts_iter = time.time()
                if prefill:
                    ts_prefill = ts_iter
                    prefill = False
                    
        self.log_tpot(ts_iter - ts_submit)
        return rid, prompt, ts_callback, ts_submit, ts_prefill, ts_iter

    async def run_engine(self):
        self.init_engine()
        self.init_done.set()
        
        tasks = []
        async def consumer():
            while True:
                try:
                    item = await asyncio.to_thread(self.inQueue.get, timeout=5)
                except queue.Empty:
                    if self.req_gen_done.is_set():
                        break
                    continue
                
                rid, ts_callback, documents = item    # ts_callback for RAG case
                sampling_params = SamplingParams(
                    temperature=1,
                    min_tokens=self.cfgs.output_len,
                    max_tokens=self.cfgs.output_len
                )
                
                task = asyncio.create_task(
                    self.streaming(documents, rid, sampling_params, ts_callback))
                tasks.append(task)
        
        while True:
            consumer_task = asyncio.create_task(consumer())
            await consumer_task 
            results = await asyncio.gather(*tasks)
            tasks.clear()
            
            for result in results:
                self.outQueue.put(result)
            
            if self.stop_flag.is_set():
                print("[VLITE] Stopping LLM Engine.")
                break

class ANNSEngine:
    def __init__(self, cfgs, reqGenDoneFlag, annsInitFlag, stopFlag,
                 searchQueue, retrieverQueue, llmQueues, statQueue, 
                 query_vectors, tpotValue):
        
        self.cfgs = cfgs
        self.reqGenDoneFlag = reqGenDoneFlag
        self.annsInitFlag = annsInitFlag
        self.stopFlag = stopFlag
        
        self.search_queue = searchQueue
        self.retriever_queue = retrieverQueue
        self.llm_queues = llmQueues
        self.stat_queue = statQueue
        
        self.query_vectors = query_vectors
        self.index = None
        self.nprobe = cfgs.search_nprobe
        self.anns_simul_time = 0.0
        
        self.tanns_tracker = []
        self.tpotValue = tpotValue

    def init_engine(self):
        if self.cfgs.search_mode is None:
            self.annsInitFlag.set()
            return
               
        slo = self.cfgs.search_slo

        if slo > 0:
            self.index = PartitionedIndex(self.cfgs, self.nprobe, self.retriever_queue)
        else:
            if self.cfgs.search_mode in ['cpu', 'hedrarag']:
                self.index = ShardedIndex(self.cfgs, self.nprobe)
            else:
                self.index = BaseGPUIndex(self.cfgs, self.nprobe)

        self.index.init_index()
        self.register_callback()
        self.annsInitFlag.set()
        
        pid = os.getpid()
        print(f"[VLITE] ANN Engine initialized with PID {pid}, Search mode={self.cfgs.search_mode}, SLO={slo}, "
              f"nprobe={self.nprobe}, and {self.cfgs.num_gpus} GPU(s).")

    # For profiling purpose
    def batch_queries(self, batch, tsArrivals, batch_sizes=None):
        if batch_sizes is None:
            n_pending_requests = self.search_queue.qsize()
            for i in range(n_pending_requests):
                rid, tsArrival, _ = self.search_queue.get()
                batch.append(self.query_vectors[rid % len(self.query_vectors)])
                tsArrivals.append(tsArrival)
            if len(batch) == 0:
                time.sleep(1e-3)
                return False
            return True
        else:
            batch_size = batch_sizes.pop(0)
            while len(batch) < batch_size:
                try:
                    rid, tsArrival, _ = self.search_queue.get(timeout=120)
                    batch.append(self.query_vectors[rid % len(self.query_vectors)])
                    tsArrivals.append(tsArrival)
                except queue.Empty:
                    if self.reqGenDoneFlag.is_set():
                        time.sleep(1)
                        return False
                    
            print(f"[VLITE] Batch size {batch_size} filled with {len(batch)} queries.")
            return True

    def run_engine(self):                    
        bid = 0
        nprobe = self.nprobe
        served_requests = 0
        batch = []
        tsArrivals = []
        tsComplete = time.time()
        k = self.cfgs.search_topk
        batch_sizes = None
        
        if self.cfgs.is_profiling and self.cfgs.search_mode:
            max_size = 32
            reps = self.cfgs.running_time
            batch_sizes = [1 for _ in range(self.cfgs.llm_workers)]
            batch_sizes.extend([i for i in range(1, max_size + 1) for _ in range(reps)])
            batch_sizes.extend([1 for _ in range(self.cfgs.llm_workers)])

        while True:
            if batch_sizes is not None and len(batch_sizes) == 0:
                print(f"[VLITE] All batches processed. Total batches: {bid}.")
                batch_sizes = None
                break
            
            if not self.batch_queries(batch, tsArrivals, batch_sizes):
                continue                
                        
            tIdling = (tsStart := time.time()) - tsComplete
            D, I = self.index.search(np.array(batch), k)
            tsComplete = time.time()
            
            self._log_stats(
                served_requests,
                bid, 
                len(batch),
                tIdling,
                tsArrivals,
                tsStart,
                tBreakdown=D[0,0:6],
                hitrates=(nprobe - D[:, 9]) / nprobe
            )

            if not self.cfgs.dispatcher:
                self.anns_callback(
                    served_requests, 
                    k, 
                    len(batch), 
                    served_requests, 
                    I.reshape(-1)
                )
                
            served_requests += len(batch)
            tsArrivals.clear()
            batch.clear()
            bid += 1
            
            if self.stopFlag.is_set():
                print("[VLITE] Stopping ANN Engine.")
                break
            
            if self.reqGenDoneFlag.is_set() and self.search_queue.empty():
                self.reset_counters()
                served_requests = 0
                bid = 0
                
    def anns_callback(self, served_request, k, batch_size, request_id, doc_ids):
        ts_callback = time.time()
        for i in range(batch_size):
            doc_id = doc_ids[i * k : (i + 1) * k]
            self.retriever_queue.put((batch_size, request_id + i, doc_id, ts_callback))
    
    def register_callback(self):
        self.index.register_callback(self.anns_callback)

    def reset_counters(self):
        self.index.reset_counters()

    def _log_stats(self, served_requests, bid, batch_size, tIdling, 
                   tsArrivals, tsStart, tBreakdown, hitrates):
        for i in range(batch_size):
            rid = served_requests + i
            self.stat_queue.put((
                rid, 
                bid,
                batch_size,
                tIdling,
                tsArrivals[i],
                tsArrivals[-1],
                tsStart,
                tBreakdown, 
                hitrates[i]
            ))

class DocRetriever:
    def __init__(self, cfgs, queues: vLiteQueues):
        self.cfgs = cfgs
        self.llm_queues = queues.llmQueue
        self.retriever_queue = queues.retrieverQueue
        self.knowledge_base = None

    def __del__(self):
        if isinstance(self.knowledge_base, plyvel.DB):
            self.knowledge_base.close()

    def run(self):
        self._setup_knowledge_base()
        num_requests = self.cfgs.total_requests + self.cfgs.warmup_requests * 2
        served_requests = 0
        
        while True:
            if self.retriever_queue.empty():
                time.sleep(1e-3)
                continue

            batch_size, rid, doc_ids, ts_callback = self.retriever_queue.get(timeout=0.001)            
            document = self.get_docs(doc_ids)
            if isinstance(document, str):
                self.llm_queues[rid % self.cfgs.llm_workers].put(
                    (rid, ts_callback, document))
            else:
                raise ValueError(f"No documents found for IDs: {doc_ids}")
            served_requests += 1
            
            if served_requests >= num_requests and self.retriever_queue.empty():
                if isinstance(self.knowledge_base, plyvel.DB):
                    self.knowledge_base.close()
                served_requests = 0
                print("[VLITE] All requests processed. Exiting document retriever.")
                break

    def get_docs(self, doc_ids):
        if isinstance(self.knowledge_base, plyvel.DB):
            doc_texts = ""
            for k in range(self.cfgs.search_topk):
                ts0 = time.time()
                doc = self.knowledge_base.get(str(doc_ids[k]).encode())
                if doc is not None:
                    doc_texts += doc.decode() + " "
                ts1 = time.time()
            return doc_texts.strip()
        else:
            return self.knowledge_base[doc_ids[0] % 1000]

    def _setup_knowledge_base(self, use_text=False):
        db_path = os.path.join(str(self.cfgs.database_dir), "text_database")
        if os.path.exists(db_path) and use_text:
            self.knowledge_base = plyvel.DB(db_path, create_if_missing=False)
        else:
            self.knowledge_base = random_prompt(self.cfgs.input_len - 1, 1000)