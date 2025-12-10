import os
import csv
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from io import StringIO
from pathlib import Path
from enum import Enum, auto
from datetime import datetime

from vliterag.configs import vLiteConfigs
from vliterag.metrics.schema import Metric, MetricStats, METRIC_CSV_NAME

class vLiteResults:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.metrics: dict[Metric, MetricStats] = {
            # ANN
            Metric.ann_queue: MetricStats(),
            Metric.ann_batch: MetricStats(),
            Metric.ann_pending: MetricStats(),
            Metric.ann_idle: MetricStats(),
            Metric.ann_search: MetricStats(),
            Metric.ann_e2e: MetricStats(),
            Metric.quantize: MetricStats(),
            Metric.lut_compute: MetricStats(),
            Metric.lut_scan: MetricStats(),
            Metric.hit_rate: MetricStats(),
            Metric.llm_queue: MetricStats(),
            Metric.prefill: MetricStats(),
            Metric.tpot: MetricStats(),
            Metric.ttft: MetricStats(),
            Metric.e2e: MetricStats(),
        }

        self.batch_sizes: dict[int, tuple[int, int]] = {}
        self.avg_batch_size = 0.0

        self.tot_time = 0.0
        self.avg_rps = 0.0
        self.avg_tps = 0.0

    def add_anns_result(self, rid, bid, batch_size, 
                        tqueue, tbatch, tpending, tidle, 
                        tsearch, ttft, tbreakdown, hr):
        m = self.metrics

        # ANN scheduler / search
        m[Metric.ann_queue].vals[rid] = tqueue
        m[Metric.ann_batch].vals[rid] = tbatch
        m[Metric.ann_pending].vals[rid] = tpending
        m[Metric.ann_idle].vals[rid] = tidle
        m[Metric.ann_search].vals[rid] = tsearch
        m[Metric.ann_e2e].vals[rid] = tsearch + tqueue
        m[Metric.ttft].vals[rid] = ttft

        # Breakdown
        m[Metric.quantize].vals[rid] = tbreakdown[0]
        m[Metric.lut_compute].vals[rid] = np.sum(tbreakdown[1:3])
        m[Metric.lut_scan].vals[rid] = np.sum(tbreakdown[3:6])
        m[Metric.hit_rate].vals[rid] = hr

        # Batch bookkeeping
        self.batch_sizes[rid] = (bid, batch_size)

    def add_llm_result(self, rid, tqueue, tpfl, tpot, te2e):
        m = self.metrics

        m[Metric.llm_queue].vals[rid] = tqueue
        m[Metric.prefill].vals[rid] = tpfl
        m[Metric.tpot].vals[rid] = tpot / self.cfgs.output_len
        m[Metric.e2e].vals[rid] = te2e

    def compute_averages(self):
        for metric_stats in self.metrics.values():
            metric_stats.compute()

        if self.tot_time > 0:
            self.avg_rps = self.cfgs.total_requests / self.tot_time
            self.avg_tps = self.avg_rps * self.cfgs.output_len

        if self.batch_sizes:
            uniq = {(bid, bsz) for (bid, bsz) in self.batch_sizes.values()}
            self.avg_batch_size = sum(bsz for (_, bsz) in uniq) / len(uniq)

    def print_results(self):
        def print_metric(label, stats):
            print(
                f"  - Avg {label}: "
                f"{stats.avg:.3f} "
                f"(P50: {stats.p50:.3f}, P90: {stats.p90:.3f})"
            )
            
        self.compute_averages()
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(
            f"[VLITE] {dt} Index {self.cfgs.index} with model {self.cfgs.model} "
            f"@ {self.cfgs.arrival_rate:.2f} rps"
        )
        print(
            f"[VLITE] {dt} LLM Workers: {self.cfgs.llm_workers}, "
            f"Search Mode: {self.cfgs.search_mode}, "
            f"SLO: {self.cfgs.search_slo}"
        )

        m = self.metrics  # shorthand
        if self.cfgs.index:
            ann_metrics = [
                (Metric.ann_queue,   "ANNQUEUE"),
                (Metric.ann_batch,   "BATCHING"),
                (Metric.ann_pending, "PENDING"),
                (Metric.ann_idle,    "CPU IDLE"),
                (Metric.ann_search,  "SEARCH"),
                (Metric.ann_e2e,     "ANNS E2E"),
            ]

            for metric, label in ann_metrics:
                if metric in m:
                    print_metric(label, m[metric])

            print(f"  - Avg Batch Size: {self.avg_batch_size:.2f}")

            if self.cfgs.search_mode == "vlite" and Metric.hit_rate in m:
                print_metric("HIT RATE", m[Metric.hit_rate])

        llm_metrics = [
            (Metric.llm_queue, "QUET"),
            (Metric.prefill,   "TPFL"),
        ]
        for metric, label in llm_metrics:
            if metric in m:
                print_metric(label, m[metric])

        if self.cfgs.index and Metric.ttft in m:
            print_metric("TTFT", m[Metric.ttft])

        for metric, label in [
            (Metric.tpot, "TPOT"),
            (Metric.e2e,  "TE2E"),
        ]:
            if metric in m:
                print_metric(label, m[metric])

        print(f"  - Total Time: {self.tot_time:.3f} seconds")
        print(f"  - Avg RPS: {self.avg_rps:.2f}")
        print(f"  - Avg TPS: {self.avg_tps:.2f}")
    
    def save_results(self):
        self.save_summary_csv()
        self.save_raw_parquet()
    
    def save_summary_csv(self):
        def cfg_to_kv(cfg):
            kv = {}
            for k, v in vars(cfg).items():
                if k.startswith("_"):
                    continue
                if v is None:
                    continue
                kv[f"{k}"] = str(v)
            return kv

        output_file = self.cfgs.result_dir / 'summary' / f"{self.cfgs.file_name}.csv"
        self.compute_averages()

        with open(output_file, "w", newline="") as f:
            w = csv.writer(f)

            # metadata header
            config_kv = cfg_to_kv(self.cfgs)
            for k, v in sorted(config_kv.items()):
                w.writerow([f'# {k}={v}'])

            w.writerow(["#"])
            w.writerow(["Metric", "Avg", "P50", "P90", "P95"])

            # metrics
            def write_metric(metric):
                stats = self.metrics[metric]
                w.writerow([
                    METRIC_CSV_NAME[metric],
                    f"{stats.avg:.6f}",
                    f"{stats.p50:.6f}",
                    f"{stats.p90:.6f}",
                    f"{stats.p95:.6f}"
                ])

            if self.cfgs.search_mode:
                for metric in [
                    Metric.ann_idle,
                    Metric.ann_batch,
                    Metric.ann_pending,
                    Metric.ann_queue,
                    Metric.ann_search,
                    Metric.ann_e2e,
                ]:
                    write_metric(metric)

                if self.cfgs.search_mode in ["cpu", "vlite"]:
                    write_metric(Metric.quantize)
                    write_metric(Metric.lut_compute)
                    write_metric(Metric.lut_scan)

                if self.cfgs.search_slo > 0:
                    write_metric(Metric.hit_rate)

                w.writerow(["Batch Size", f"{self.avg_batch_size:.6f}"])

            write_metric(Metric.llm_queue)
            write_metric(Metric.prefill)
            write_metric(Metric.tpot)
            write_metric(Metric.e2e)

            w.writerow(["Total Time", f"{self.tot_time:.6f}"])
            w.writerow(["Avg RPS", f"{self.avg_rps:.6f}"])

        print(f"[VLITE] Results saved to {output_file}")

    def read_summary_csv(self, file_path):
        rows = []
        meta = {}

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("#"):
                    # # key=value
                    kv = line[1:].strip()
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        meta[k.strip()] = v.strip()
                    continue

                if line.startswith('"#') and line.endswith('"'):
                    continue

                rows.append(line)

        for k, v in meta.items():
            if not hasattr(self.cfgs, k):
                continue
            try:
                if v.startswith("{") or v.startswith("["):
                    val = ast.literal_eval(v)
                elif v.lower() in ("true", "false"):
                    val = v.lower() == "true"
                else:
                    val = float(v) if "." in v else int(v)
            except Exception:
                val = v

            setattr(self.cfgs, k, val)

        if not rows:
            raise ValueError(f"No CSV body found in {file_path}")

        df = pd.read_csv(StringIO("\n".join(rows)))
        return df

    def save_raw_parquet(self, output_file=None):
        def cfg_to_parquet_metadata(cfg, prefix="cfg."):
            meta = {}
            for k, v in vars(cfg).items():
                if k.startswith("_"):
                    continue
                if v is None:
                    continue
                meta[f"{prefix}{k}"] = str(v)
            return meta
        
        if output_file is None:
            output_file = self.cfgs.result_dir / 'raw' / f"{self.cfgs.file_name}_raw.parquet"

        rows = []
        m = self.metrics
        num_rows = len(m[Metric.llm_queue].vals)
        for i in range(num_rows):
            row = {}

            # basic id
            bid, bsize = self.batch_sizes.get(i, (-1, -1))
            row["rid"] = i
            row["batch_id"] = bid
            row["batch_size"] = bsize

            # ann
            if self.cfgs.search_mode:
                row.update({
                    "ann_idle": m[Metric.ann_idle].vals.get(i, np.nan),
                    "ann_batch": m[Metric.ann_batch].vals.get(i, np.nan),
                    "ann_pending": m[Metric.ann_pending].vals.get(i, np.nan),
                    "ann_queue": m[Metric.ann_queue].vals.get(i, np.nan),
                    "ann_search": m[Metric.ann_search].vals.get(i, np.nan),
                    "ann_e2e": m[Metric.ann_e2e].vals.get(i, np.nan),
                    "ttft": m[Metric.ttft].vals.get(i, np.nan),
                })

            # ann breakdown
            if self.cfgs.search_mode in ["cpu", "vlite"]:
                row.update({
                    "quantize": m[Metric.quantize].vals.get(i, np.nan),
                    "lut_compute": m[Metric.lut_compute].vals.get(i, np.nan),
                    "lut_scan": m[Metric.lut_scan].vals.get(i, np.nan),
                })

                if self.cfgs.search_slo > 0:
                    row["hit_rate"] = m[Metric.hit_rate].vals.get(i, np.nan)

            # llm
            row.update({
                "llm_queue": m[Metric.llm_queue].vals.get(i, np.nan),
                "prefill": m[Metric.prefill].vals.get(i, np.nan),
                "tpot": m[Metric.tpot].vals.get(i, np.nan),
                "e2e": m[Metric.e2e].vals.get(i, np.nan),
            })

            rows.append(row)

        df = pd.DataFrame(rows)

        # parquet metadata 
        meta = cfg_to_parquet_metadata(self.cfgs)
        table = pa.Table.from_pandas(df)
        table = table.replace_schema_metadata({
            k.encode(): v.encode() for k, v in meta.items()
        })
        
        pq.write_table(table, output_file)
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[VLITE] {dt} Raw stats saved to {output_file}")

    def read_raw_parquet(self, file_path=None):
        """
        Restore rid-level raw stats + cfg metadata from parquet.
        Returns: pd.DataFrame
        """
        if file_path is None:
            file_path = self.cfgs.result_dir / "raw" / f"{self.cfgs.file_name}_raw.parquet"

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Raw parquet not found: {file_path}")

        df = pd.read_parquet(file_path)

        pf = pq.ParquetFile(file_path)
        meta = {
            k.decode(): v.decode()
            for k, v in (pf.metadata.metadata or {}).items()
        }

        _apply_cfg_metadata(self.cfgs, meta)
        return df
    
    def result_exists(self):
        summary = self.cfgs.result_dir / 'summary' / f"{self.cfgs.file_name}.csv"
        raw = self.cfgs.result_dir / 'raw' / f"{self.cfgs.file_name}_raw.parquet"
        return summary.exists() and raw.exists()
    
def _apply_cfg_metadata(cfg, meta: dict, prefix="cfg."):
    for k, v in meta.items():
        if not k.startswith(prefix):
            continue

        attr = k[len(prefix):]
        if not hasattr(cfg, attr):
            continue

        try:
            val = float(v)
            if val.is_integer():
                val = int(val)
        except ValueError:
            if v.lower() in ("true", "false"):
                val = v.lower() == "true"
            else:
                val = v

        setattr(cfg, attr, val)