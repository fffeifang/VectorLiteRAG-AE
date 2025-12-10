import numpy as np
from enum import Enum, auto

class Metric(Enum):
    # ANN
    ann_queue = auto()
    ann_batch = auto()
    ann_pending = auto()
    ann_idle = auto()
    ann_search = auto()
    ann_e2e = auto()

    # ANN breakdown
    quantize = auto()
    lut_compute = auto()
    lut_scan = auto()
    hit_rate = auto()

    # LLM
    llm_queue = auto()
    prefill = auto()
    tpot = auto()

    # Combined
    ttft = auto()
    e2e = auto()

class MetricStats:
    def __init__(self):
        self.vals = dict()
        self.avg = 0.0
        self.p50 = 0.0
        self.p90 = 0.0
        self.p95 = 0.0
    
    def compute(self):
        if self.vals:
            self.avg = np.mean(list(self.vals.values()))
            self.p50 = np.percentile(list(self.vals.values()), 50)
            self.p90 = np.percentile(list(self.vals.values()), 90)
            self.p95 = np.percentile(list(self.vals.values()), 95)
        else:
            self.avg = self.p50 = self.p90 = self.p95 = 0.0
    
METRIC_CSV_NAME = {
    Metric.ann_idle: "Idle",
    Metric.ann_queue: "Queuing",
    Metric.ann_batch: "Batching",
    Metric.ann_pending: "Pending",
    Metric.ann_search: "Search",
    Metric.ann_e2e: "ANNS E2E",

    Metric.quantize: "Quantize",
    Metric.lut_compute: "LUT Computed",
    Metric.lut_scan: "LUT Scan",
    Metric.hit_rate: "Hit Rate",

    Metric.llm_queue: "LLM Queuing",
    Metric.prefill: "Prefill",
    Metric.tpot: "TPOT",

    Metric.ttft: "TTFT",
    Metric.e2e: "End2End",
}