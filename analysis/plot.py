import os
import copy
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from copy import deepcopy
from pathlib import Path
from matplotlib.ticker import FuncFormatter

from configs.loader import load_index, load_model, load_all_models
from vliterag.args  import parse_args
from vliterag.configs import vLiteConfigs
from vliterag.results import vLiteResults
from vliterag.profiler import LatencyEstimator, HitRateEstimator

from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib import patches

PRJ_ROOT = Path(__file__).resolve().parents[1]
(PRJ_ROOT / 'figures').mkdir(parents=True, exist_ok=True)

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size'] = 10

INDEXES = ['wikiall', 'orcas1k', 'orcas2k']
MODELS = ['llama8b', 'qwen32b', 'llama70b']

def parse_csvs(index, model, num_gpus, mode, tag='main'):
    path = PRJ_ROOT / 'results' / index / model / f"{num_gpus}gpus" / mode / 'summary'
    
    ret = []
    cfg = vLiteConfigs(is_plotting=True)

    if not path.exists():
        return ret

    for file in path.iterdir():
        if not file.name.endswith(".csv"):
            continue

        res = vLiteResults(cfg)
        df = res.read_summary_csv(file) 
        cfg = res.cfgs
        if not all([
            cfg.index == index,
            cfg.model == model,
            cfg.num_gpus == num_gpus,
            not cfg.is_profiling,
            cfg.search_mode == mode,
            cfg.file_tag == tag,
        ]):
            continue
        ret.append((deepcopy(cfg), df))
    return ret

def parse_files(index, model, num_gpus, mode, tag='main'):
    path = PRJ_ROOT / 'results' / index / model / f"{num_gpus}gpus" / mode / 'raw'
    
    ret = []
    cfg = vLiteConfigs(is_plotting=True)

    if not path.exists():
        return ret

    for file in path.iterdir():
        if not file.name.endswith(".parquet"):
            continue

        res = vLiteResults(cfg)
        df = res.read_raw_parquet(file)
        cfg = res.cfgs
        if not all([
            cfg.index == index,
            cfg.model == model,
            cfg.num_gpus == num_gpus,
            not cfg.is_profiling,
            cfg.search_mode == mode,
            cfg.file_tag == tag,
        ]):
            continue
        ret.append((deepcopy(cfg), df))
    return ret

def run_perf_model(cfg):
    nprobe = cfg.search_nprobe
    latency_estimator = LatencyEstimator(32, nprobe)
    latency_estimator.load_latency_data(cfg)
    
    if cfg.search_mode == 'vlite':
        hitrate_estimator = HitRateEstimator(nprobe)
        hitrate_estimator.load_centroids_data(cfg)

        centroid_dir = Path(cfg.database_dir) / cfg.index / cfg.model / f"{cfg.num_gpus}gpus" / 'shards'
        mtd_path = centroid_dir / f"{cfg.search_slo}ms_meta_{nprobe}.txt"

        with open(mtd_path, 'r') as f:
            for line in f:
                if "mean hitrate" in line:
                    exp_min_hitrate = float(line.strip().split(":")[1].strip())
                    break
        
        hit_rates = np.array([
            hitrate_estimator.compute_min_hitrate(bs, exp_min_hitrate) for bs in range(1, 33)
        ])
    else:
        hit_rates = np.zeros(32)

    latency_estimator.hitrate_data = hit_rates
    pred_search_time = np.array(
        [latency_estimator.estimate_latency(bs)[1] for bs in range(1, 33)])
    
    return pd.DataFrame({
        "batch_size": range(1, 33),
        "hit_rate": latency_estimator.hitrate_data,
        "ann_search": pred_search_time
    })

def plot_perf_model(axes):
    model_list = list(load_all_models().keys())
    index_list = list(load_index().keys())
    
    CMAP = {'wikiall': '#0071c5', 
            'orcas1k': '#165b0e', 
            'orcas2k': '#FF912A'}

    MODEL = 'llama8b'
    MODE = 'vlite'
    TAG = 'main'
    NUM_GPUS = 8
    
    ax1, ax2 = axes
    
    for index in INDEXES:
        du_tups = parse_files(index, MODEL, NUM_GPUS, MODE, TAG)
        dfs = []
        cfg = None
        for config, df in du_tups:
            if not all([
                config.input_len == 1024,
                config.output_len == 256,
                config.num_gpus == NUM_GPUS
            ]):
                continue
            dfs.append(df.groupby('batch_id').agg(
                batch_size=('batch_size', 'mean'),
                ann_search=('ann_search', 'max'),
                hit_rate=('hit_rate', 'min')
            ))
            cfg = deepcopy(config)
            
        if len(dfs) == 0:
            continue
        
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values(by='batch_size', ascending=True).reset_index(drop=True)
        df = df.groupby('batch_size').filter(lambda x: len(x) >= 20)
        df = df.groupby('batch_size').mean().reset_index()

        cfg.is_profiling = True
        df_pred = run_perf_model(cfg)

        max_batch_size = df.index.max()
        df_pred = df_pred[df_pred.index <= max_batch_size]
        
        ax1.plot(df['batch_size'], df['ann_search'] * 1000, label=index,
                  marker='^', color=CMAP[index])
        ax1.plot(df_pred['batch_size'], df_pred['ann_search'] * 1000, label=index,
                marker='s', color=CMAP[index], linestyle=':', alpha=0.75)
        
        ax2.plot(df['batch_size'], df['hit_rate'], label=index,
                  marker='^', color=CMAP[index])
        ax2.plot(df_pred['batch_size'], df_pred['hit_rate'], label=index,
                marker='s', color=CMAP[index], linestyle='--', alpha=0.75)
        
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Search Time (ms)")
        ax1.grid(axis='y', linestyle='--', alpha=0.4)

        ax2.set_xlabel("Batch Size")
        ax2.set_ylabel("Tail Query Hit Rate")
        ax2.grid(axis='y', linestyle='--', alpha=0.4)
        ax2.legend(frameon=False, fontsize=10)

        ax1.set_xticks([1, 4, 7, 10, 13])
        ax2.set_xticks([1, 4, 7, 10, 13])

def get_slo_attainments(slo, df):
    attainment = (df["ttft"] <= slo).sum() / len(df)
    return attainment

def plot_slo_attainment(index, model, axes):

    MODES = ['cpu', 'all-gpu', 'ded-gpu', 'vlite']    
    GPU_TYPE = "L40S" if model == "llama8b" else "H100"
    INPUT = 1024
    OUTPUT = 256
    TAG = 'main'
    NUM_GPUS = 8
    
    CMAP = {'cpu': '#0071c5', 
            'ded-gpu': '#165b0e', 
            'all-gpu': '#76b900',
            'vlite': '#FF912A'
    }
    
    MMAP = {
        'cpu': 'D',
        'ded-gpu': 's',
        'all-gpu': 'v',
        'vlite': 'o'    
    }
    
    search_slo = load_index()[index]['slo'] / 1000
    slo_list = load_model(model)['slo'][GPU_TYPE]
    slo_entry = next(x for x in slo_list if x['input'] == INPUT and x['output'] == OUTPUT)
    prefill_slo = slo_entry['values'][f"ngpu={NUM_GPUS}"]
    
    tput_list = load_model(model)['tput_ceiling'][GPU_TYPE]
    tput_entry = next(x for x in tput_list if x['input'] == INPUT and x['output'] == OUTPUT)
    tput_ceiling = tput_entry['values'][f"ngpu={NUM_GPUS}"]
    
    slo = search_slo + prefill_slo

    for mode in MODES:
        data = {}
        df_tups = parse_files(index, model, 8, mode, 'main')
        for configs, df in df_tups:
            if not all([
                configs.input_len == 1024,
                configs.output_len == 256,
                configs.num_gpus == NUM_GPUS
            ]):
                continue
            if index == 'orcas1k' and configs.search_slo != 200:
                continue
            arrival_rate = configs.arrival_rate
            attainment_rate = get_slo_attainments(slo, df)
            if arrival_rate in data:
                print(f"Warning multiple data values captured at "
                    f"{index}, {model}, {mode} with {configs}")
            data[arrival_rate] = attainment_rate

        data = {k: v for k, v in sorted(data.items())}
        axes.plot(data.keys(), data.values(), label=mode, 
                   markersize=5, marker=MMAP[mode], color=CMAP[mode])

    xmin, xmax = axes.get_xlim()

    axes.set_ylim(0,)
    axes.set_xlim(xmin, xmax)

    axes.hlines(xmin=xmin, xmax=xmax, y=0.90, color='gray', linestyle='--')
    axes.vlines(x=tput_ceiling, ymin=0, ymax=0.9, color='gray', linestyle='--')
    
    axes.grid(axis='y', linestyle='--', alpha=0.4)
    axes.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))

    axes.set_xlabel("Arrival Rate (req/s)")    
    axes.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # hiding redundant information in whole figure
    axes.set_ylabel("")
    
    # set xticks to integer values
    if index !="orcas2k":
        axes.set_xlabel("")
        axes.set_xticklabels([])
    if model == "llama8b":
        ylabels = {"wikiall": "Wiki-All", "orcas1k": "SLO Attainment\nORCAS 1K", "orcas2k": "ORCAS 2K"}
        axes.set_ylabel(f"{ylabels[index]}")
    else:
        axes.set_yticklabels([])
    if index == "wikiall":
        titles = {"llama8b": "Llama3-8B", "qwen32b": "Qwen3-32B", "llama70b": "Llama3-70B"}
        axes.set_title(f"{titles[model]}", fontsize=10)
    return axes

def plot_e2e_latency(index, model, axes):
    
    MODES = ['cpu', 'all-gpu', 'ded-gpu', 'vlite']    
    GPU_TYPE = "L40S" if model == "llama8b" else "H100"
    TAG = 'main'
    NUM_GPUS = 8
    
    CMAP = {'cpu': '#0071c5', 
            'ded-gpu': '#165b0e', 
            'all-gpu': '#76b900',
            'vlite': '#FF912A'
    }
    
    MMAP = {
        'cpu': 'D',
        'ded-gpu': 's',
        'all-gpu': 'v',
        'vlite': 'o'    
    }
    
    for mode in MODES:
        data = {}
        df_tups = parse_files(index, model, 8, mode, 'main')
        for configs, df in df_tups:
            if not all([
                configs.input_len == 1024,
                configs.output_len == 256,
                configs.num_gpus == NUM_GPUS
            ]):
                continue
            if index == 'orcas1k' and configs.search_slo != 200:
                continue
            arrival_rate = configs.arrival_rate
            avg_e2e = df['e2e'].mean()
            if arrival_rate in data:
                print(f"Warning multiple data values captured at "
                    f"{index}, {model}, {mode} with {configs}")
            data[arrival_rate] = avg_e2e
    
        data = {k: v for k, v in sorted(data.items())}
        axes.plot(data.keys(), data.values(), label=mode, 
                   markersize=5, marker=MMAP[mode], color=CMAP[mode])
                   
    axes.set_xlabel("Arrival Rate (req/s)")
    axes.set_ylabel("Latency (s)")
    axes.set_ylim(0, 60)
    axes.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.4)
    
    # hiding redundant information in whole figure
    axes.set_ylabel("")
    if index !="orcas2k":
        axes.set_xlabel("")
        axes.set_xticklabels([])
    if model == "llama8b":
        labels = {"wikiall": "Wiki-All", "orcas1k": "ORCAS 1K", "orcas2k": "ORCAS 2K"}
        ylabel = f"{labels[index]}"
        if index == "orcas1k":
            ylabel = "End-to-End Latency (s)\n" + ylabel
        axes.set_ylabel(f"{ylabel}", fontsize=10)
    else:
        axes.set_yticklabels([])
    if index == "wikiall":
        titles = {"llama8b": "Llama3-8B", "qwen32b": "Qwen3-32B", "llama70b": "Llama3-70B"}
        axes.set_title(f"{titles[model]}", fontsize=10)
                       
    return axes

def plot_figure_11():
    mainFig, mainAxes = plt.subplots(3, 6, figsize=(16, 6), constrained_layout=True)

    for iv, index in enumerate(INDEXES):
        for im, model in enumerate(MODELS):
            plot_slo_attainment(index, model, mainAxes[iv, im])

    for iv, index in enumerate(INDEXES):
        for im, model in enumerate(MODELS):
            plot_e2e_latency(index, model, mainAxes[iv, im + 3])

    plt_engine = mainFig.get_layout_engine()
    plt_engine.set(rect=(0, 0.05, 1, 0.95))

    handles, labels = mainAxes[0, 0].get_legend_handles_labels()
    mainFig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=4,
        fontsize=10,
        frameon=False
    )

    plt.savefig(f"{str(PRJ_ROOT)}/figures/figure_11.pdf")
    plt.close(mainFig)

def plot_figure_12():
    
    INDEXES = ['wikiall', 'orcas1k']
    MODEL = 'qwen32b'
    MODES = ['ded-gpu', 'all-gpu', 'vlite', 'cpu']
    ARRIVAL_RATES = [19, 32, 38]
    GPU_TYPE = 'H100'
    NUM_GPUS = 8
    
    CMAP = {
        'cpu': ['whitesmoke', '#0071c5', "#555555"], 
        'ded-gpu': ['whitesmoke', "#4a7007", "#555555"],
        'all-gpu': ['whitesmoke', "#45c037", '#555555'],
        'vlite': ['whitesmoke', '#FF912A', "#555555"]
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.67), sharey=True)
    
    for axis_id, index in enumerate(INDEXES):
        data = {}
        ax = axes[axis_id]
        for mode in MODES:
            df_tups = parse_files(index, MODEL, NUM_GPUS, mode, 'main')
            dfs = []
            for config, df in df_tups:
                if config.arrival_rate in ARRIVAL_RATES:
                    dfs.append(df)
            
            if len(dfs) == 0:
                continue 
            
            df = pd.concat(dfs, ignore_index=True)
            data[mode] = df.mean()
            
        n_rates = len(ARRIVAL_RATES)
        n_modes = len(MODES)
        
        bar_width = 0.75
        pos = np.arange(1, n_rates * n_modes + 1)
        vl_pos = [(pos[i-1] + pos[i]) / 2 for i, v in enumerate(pos) if i > 0 and i % n_modes == 0]
        al_pos = [(pos[i * n_modes] + pos[(i + 1) * n_modes + - 1]) / 2 for i in range(n_rates)]

        for i, rate in enumerate(ARRIVAL_RATES):
            for j, mode in enumerate(MODES):
                if mode not in data:
                    continue 
                
                bot = 0
                df = data[mode]
                ax.bar(pos[i * n_modes + j], df['ann_queue'],
                   width=bar_width, bottom=bot,label=f"Queuing Delay" if i + j == 0 else None, 
                   color=CMAP[mode][0], hatch='//', edgecolor="#010201")
                bot += df['ann_queue']
                
                ax.bar(pos[i * n_modes + j], df['ann_search'],
                    width=bar_width, bottom=bot, label=mode if i == 0 else None,
                    color=CMAP[mode][1], edgecolor='#010201')
                bot += df['ann_search']
                
                ax.bar(pos[i * n_modes + j], df['prefill'],
                    width=bar_width, bottom=bot, label=f"Prefill" if i + j == n_rates + n_modes - 2 else None,
                    color=CMAP[mode][2], hatch='..', edgecolor="#010201")
        
        for vl in vl_pos:
            ax.axvline(x=vl, color='black', linestyle='--', linewidth=0.5)  
    
        latency_lim = 0.4
        ax.set_xticks(al_pos)
        ax.set_xticklabels([f"{rate}" for rate in ARRIVAL_RATES])
        ax.set_ylim(0, latency_lim)
        ax.set_ylabel("Latency (s)" if index == "wikiall" else None)

        title = "Wiki-All" if index == "wikiall" else "ORCAS 1K"
        ax.set_xlabel(f"{title} - Arrival Rate (req/s)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=8.5, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{str(PRJ_ROOT)}/figures/figure_12.pdf')
    plt.close(fig)
    
def plot_figure_10():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.12))
    plot_perf_model(axes)
    plt.tight_layout()
    plt.savefig(f'{str(PRJ_ROOT)}/figures/figure_10.pdf')
    plt.close()

def plot_figure_14():
    """ Ablation study: Dispatcher on/off """

    MODEL = "llama8b"
    INDEX = "orcas2k"
    NUM_GPUS = 8
    MODE = 'vlite'
    ARRIVAL_RATES = [24, 32, 41]
    TAG = 'dispatcher'

    onAvgSearch, offAvgSearch = {}, {}
    onP90Search, offP90Search = {}, {}
    onBatchSize, offBatchSize = {}, {}

    df_tups = parse_csvs(INDEX, MODEL, NUM_GPUS, MODE, TAG)
    for cfg, df in df_tups:
        aps = cfg.arrival_rate
        if aps not in ARRIVAL_RATES:
            continue

        is_on = cfg.dispatcher

        if is_on:
            onAvgSearch[aps]  = df.loc['Search', 'Avg'] * 1000
            onP90Search[aps]  = df.loc['Search', 'P90'] * 1000
            onBatchSize[aps]  = df.loc['Batch Size', 'Metric']
        else:
            offAvgSearch[aps] = df.loc['Search', 'Avg'] * 1000
            offP90Search[aps] = df.loc['Search', 'P90'] * 1000
            offBatchSize[aps] = df.loc['Batch Size', 'Metric']
    
    def vals(d):
        return [d[r] for r in ARRIVAL_RATES if r in d]

    fig, ax = plt.subplots(1, 2, figsize=(8, 8/3))
    xpos = np.arange(len(ARRIVAL_RATES))
    width = 0.3

    ax[0].bar(xpos - width/2, vals(onAvgSearch), width,
              label='Dispatcher On', color='#FF912A', edgecolor='black')
    ax[0].bar(xpos + width/2, vals(offAvgSearch), width,
              label='Dispatcher Off', color="#555555", edgecolor='black')

    ax0_tw = ax[0].twinx()
    ax0_tw.scatter(xpos - width/2, vals(onBatchSize),
                   color="#FF2A2A", marker='o')
    ax0_tw.scatter(xpos + width/2, vals(offBatchSize),
                   color="#6F0A0A", marker='^')

    ax[0].set_ylabel('Avg Search Time (ms)')
    ax0_tw.set_ylabel('Batch Size')
    ax[0].set_xlabel('Arrival Rate (req/s)')
    ax[0].set_ylim(100,)
    ax[0].set_xticks(xpos)
    ax[0].set_xticklabels(ARRIVAL_RATES)
    ax[0].legend(frameon=False, fontsize=9, loc='upper left')

    for x, y1, y2 in zip(xpos, vals(onBatchSize), vals(offBatchSize)):
        ax0_tw.text(x - width/2, y1, f'{y1:.1f}',
                    ha='center', va='bottom', fontsize=8)
        ax0_tw.text(x + width/2, y2, f'{y2:.1f}',
                    ha='center', va='bottom', fontsize=8)

    ax[1].bar(xpos - width/2, vals(onP90Search), width,
              label='Dispatcher On', color="#FF912A", edgecolor='black')
    ax[1].bar(xpos + width/2, vals(offP90Search), width,
              label='Dispatcher Off', color='#555555', edgecolor='black')

    ax[1].set_ylabel('P90 Search Time (ms)')
    ax[1].set_ylim(100,)
    ax[1].set_xlabel('Arrival Rate (req/s)')
    ax[1].set_xticks(xpos)
    ax[1].set_xticklabels(ARRIVAL_RATES)

    for a in ax:
        a.grid(axis='y', linestyle='--', alpha=0.85)

    plt.tight_layout()
    plt.savefig(f'{PRJ_ROOT}/figures/figure_14.pdf', dpi=300)
    plt.close()
    
def plot_figure_15():
    """ Ablation study: different input/output lengths """

    inlen_lists = [512, 1024, 2048]
    outlen_lists = [128, 256, 512]
    MODELS = ['llama8b', 'llama70b']
    MODES = ['all-gpu', 'vlite', 'cpu']
    INDEX = 'orcas2k'
    NUM_GPUS = 8
    GPU_TYPE = 'L40S'

    fig, ax = plt.subplots(2, 2, figsize=(8, 5), sharey=True)
    ax = ax.flatten()

    colors = {
        "cpu": ["#9bbdfb", "#4d7cd1", "#061E4A"],
        "all-gpu": ["#92c638", "#76b900", "#203200"],
        "vlite": ["#FFBD80", "#FF912A", "#361C03"]
    }
    markers = {"cpu": 'o', "all-gpu": 's', "vlite": '^'}
    linetypes = ["dashed", "solid", "dotted"]

    for i, model in enumerate(MODELS):
        # ---- Prefill SLO line (fixed reference) ----
        slo_entry = next(
            x for x in load_model(model)['slo'][GPU_TYPE]
            if x['input'] == 1024 and x['output'] == 256
        )
        prefill_slo = slo_entry['values'][f"ngpu={NUM_GPUS}"] * 1000

        ax_l = ax[2 * i]
        min_aps, max_aps = float('inf'), 0

        for j, inlen in enumerate(inlen_lists):
            for mode in MODES:
                aps_to_p90 = {}

                for cfg, df in parse_files(INDEX, model, NUM_GPUS, mode):
                    if cfg.input_len == inlen and cfg.output_len == 256:
                        aps_to_p90[cfg.arrival_rate] = (
                            df['ttft'].quantile(0.90) * 1000
                        )

                if not aps_to_p90:
                    continue

                aps_to_p90 = dict(sorted(aps_to_p90.items()))
                ax_l.plot(
                    aps_to_p90.keys(),
                    aps_to_p90.values(),
                    marker=markers[mode],
                    linestyle=linetypes[j],
                    color=colors[mode][2 - j],
                    label=f'{mode} {inlen}' if j == 0 else f'{inlen}'
                )

                min_aps = min(min_aps, min(aps_to_p90.keys()))
                max_aps = max(max_aps, max(aps_to_p90.keys()))

        ax_l.axhline(y=prefill_slo + 300, color='red', linestyle='--', linewidth=1)
        ax_l.set_ylabel(f'{model}\nP90 TTFT (ms)')
        ax_l.set_ylim(0, 1000)
        ax_l.grid(axis='y', linestyle='--', alpha=0.6)
        ax_l.set_xticks(
            np.linspace(min_aps, max_aps, 5, dtype=int)
        )

        ax_r = ax[2 * i + 1]
        min_aps, max_aps = float('inf'), 0

        for j, outlen in enumerate(outlen_lists):
            for mode in MODES:
                aps_to_p90 = {}

                for cfg, df in parse_files(INDEX, model, NUM_GPUS, mode):
                    if cfg.input_len == 1024 and cfg.output_len == outlen:
                        aps_to_p90[cfg.arrival_rate] = (
                            df['ttft'].quantile(0.90) * 1000
                        )

                if not aps_to_p90:
                    continue

                aps_to_p90 = dict(sorted(aps_to_p90.items()))
                ax_r.plot(
                    aps_to_p90.keys(),
                    aps_to_p90.values(),
                    marker=markers[mode],
                    linestyle=linetypes[j],
                    color=colors[mode][2 - j],
                    label=f'{mode} {outlen}' if j == 0 else f'{outlen}'
                )

                min_aps = min(min_aps, min(aps_to_p90.keys()))
                max_aps = max(max_aps, max(aps_to_p90.keys()))

        ax_r.axhline(y=prefill_slo + 300, color='red', linestyle='--', linewidth=1)
        ax_r.set_ylim(0, 1000)
        ax_r.grid(axis='y', linestyle='--', alpha=0.6)
        ax_r.set_xticks(
            np.linspace(min_aps, max_aps, 5, dtype=int)
        )

    ax[0].set_title('Input Length Ablation', fontsize=10)
    ax[1].set_title('Output Length Ablation', fontsize=10)
    ax[2].set_xlabel('Arrival Rate (req/s)')
    ax[3].set_xlabel('Arrival Rate (req/s)')

    handles_input = [
        Line2D([], [], color='#4d7cd1', marker='o', label='CPU Only'),
        Line2D([], [], color="#2C2C2C", label='2048/256', linewidth=2, linestyle='dashed'),
        Line2D([], [], color='#FF912A', marker='^', label='vLiteRAG'),
        Line2D([], [], color="#727272", label='1024/256', linewidth=2, linestyle='solid'),
        Line2D([], [], color='#76b900', marker='s', label='ALL-GPU'),
        Line2D([], [], color="#D3D3D3", label='512/256', linewidth=2, linestyle='dotted'),
    ]

    legend_input = fig.legend(
        handles_input,
        [h.get_label() for h in handles_input],
        loc='lower left',
        fontsize=8,
        ncol=3,
        bbox_to_anchor=(0.11, 0),
        frameon=False
    )

    handles_output = [
        Line2D([], [], color='#4d7cd1', marker='o', label='CPU Only'),
        Line2D([], [], color="#2C2C2C", label='1024/512', linewidth=2, linestyle='dashed'),
        Line2D([], [], color='#FF912A', marker='^', label='vLiteRAG'),
        Line2D([], [], color="#727272", label='1024/256', linewidth=2, linestyle='solid'),
        Line2D([], [], color='#76b900', marker='s', label='ALL-GPU'),
        Line2D([], [], color="#D3D3D3", label='1024/128', linewidth=2, linestyle='dotted'),
    ]

    legend_output = fig.legend(
        handles_output,
        [h.get_label() for h in handles_output],
        loc='lower right',
        fontsize=8,
        ncol=3,
        bbox_to_anchor=(0.97, 0),
        frameon=False
    )

    fig.add_artist(legend_input)

    plt.tight_layout(rect=[0, 0.09, 1, 1])
    plt.savefig(f'{str(PRJ_ROOT)}/figures/figure_15.pdf', dpi=300)
    plt.close()
    
def plot_figure_16():
    """ Ablation study: different search SLO levels """

    GPU_TYPE = "H100"
    NUM_GPUS = 8
    INDEX = 'orcas1k'
    MODEL = 'qwen32b'
    MODES = ['cpu', 'all-gpu', 'vlite']
    SLOS = [100, 150, 200, 250]

    CMAP = {
        "cpu": "#4d7cd1",
        "all-gpu": "#76b900",
        "vlite": "#FF912A",
        "vlite_90": "#A05510"
    }

    fig, ax = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True)
    ax = ax.flatten()

    slo_entry = next(
        x for x in load_model(MODEL)['slo'][GPU_TYPE]
        if x['input'] == 1024 and x['output'] == 256
    )
    prefill_slo = slo_entry['values'][f"ngpu={NUM_GPUS}"] * 1000

    global_min_aps, global_max_aps = float('inf'), 0

    for i, search_slo in enumerate(SLOS):
        for mode in MODES:
            aps_to_quantiles = {}

            for cfg, df in parse_files(INDEX, MODEL, NUM_GPUS, mode):
                if cfg.search_slo != search_slo:
                    continue

                aps_to_quantiles[cfg.arrival_rate] = (df['ttft'].quantile([0.5, 0.9, 0.95]) * 1000)

            if not aps_to_quantiles:
                continue

            aps_to_quantiles = dict(sorted(aps_to_quantiles.items()))
            aps = list(aps_to_quantiles.keys())
            p90 = [v[1] for v in aps_to_quantiles.values()]
            p95 = [v[2] for v in aps_to_quantiles.values()]

            global_min_aps = min(global_min_aps, min(aps))
            global_max_aps = max(global_max_aps, max(aps))

            ax[i].plot(aps, p95, marker='o', color=CMAP[mode], alpha=1.0 if mode == 'vlite' else 0.4, label=mode if i == 0 else None)

            if mode == 'vlite':
                ax[i].plot(aps, p90, marker='x', linestyle='--', color=CMAP['vlite_90'], label='vLiteRAG P90' if i == 0 else None)

        ax[i].set_title(f"Search SLO: {search_slo} ms", fontsize=10)
        ax[i].axhline(y=search_slo + prefill_slo, color='red', linestyle='--', linewidth=1)
        ax[i].set_ylim(0, 1000)
        ax[i].grid(axis='y', linestyle='--', alpha=0.4)

        if i >= 2:
            ax[i].set_xlabel('Arrival Rate (req/s)')
        if i % 2 == 0:
            ax[i].set_ylabel('P95 TTFT (ms)')

    ax[0].legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{str(PRJ_ROOT)}/figures/figure_16.pdf', dpi=300)
    plt.close()
    
def plot_figure_17():
    """ Ablation study: GPU Number per node """

    INDEX = 'orcas2k'
    MODEL = 'qwen32b'
    GPU_TYPE = 'H100'
    MODES = ['cpu', 'all-gpu', 'vlite']
    NUM_GPUS = [4, 6, 8]
    CMAP = {"cpu": "#4d7cd1", "all-gpu": "#76b900", "vlite": "#FF912A"}
    SEARCH_SLO = 300  # ms

    fig, ax = plt.subplots(1, 2, figsize=(8, 8/3))
    ax = ax.flatten()

    markers = ['o', 's', '^']
    linestyles = ['-', '--', '-.']

    for i, ngpu in enumerate(NUM_GPUS):
        # Prefill SLO
        slo_entry = next(
            x for x in load_model(MODEL)['slo'][GPU_TYPE]
            if x['input'] == 1024 and x['output'] == 256
        )
        prefill_slo = slo_entry['values'][f"ngpu={ngpu}"] * 1000
        slo_threshold = SEARCH_SLO + prefill_slo

        data_atm_all = []

        for mode in MODES:
            data_atm = {}
            data_e2e = {}

            for config, df in parse_files(INDEX, MODEL, ngpu, mode):
                if config.search_slo != SEARCH_SLO:
                    continue

                aps = config.arrival_rate
                data_atm[aps] = get_slo_attainments(slo_threshold, df)
                data_e2e[aps] = df['e2e'].mean()

            if not data_atm or not data_e2e:
                continue

            data_atm = dict(sorted(data_atm.items()))
            data_e2e = dict(sorted(data_e2e.items()))
            data_atm_all.extend(data_atm.keys())

            ax[0].plot(data_atm.keys(), data_atm.values(), marker=markers[i], linestyle=linestyles[i], color=CMAP[mode], alpha=0.8)
            ax[1].plot(data_e2e.keys(), data_e2e.values(), marker=markers[i], linestyle=linestyles[i], color=CMAP[mode], alpha=0.8)

        if data_atm_all:
            max_aps = max(data_atm_all)
            ax[0].axvline(max_aps, ymin=0, ymax=0.85,
                          color='gray', linestyle='--', alpha=0.8)
            ax[0].text(max_aps - 1.5, 0.5,
                       f'{ngpu} GPUs Cap',
                       rotation=270,
                       fontsize=8,
                       color='gray')

    ax[0].set_xlabel('Arrival Rate (req/s)')
    ax[0].set_ylabel('SLO Attainment')
    ax[0].set_ylim(0, 1.05)
    ax[0].axhline(0.9, linestyle='--', alpha=0.8, color='gray')
    ax[0].grid(axis='y', linestyle='--', alpha=0.4)

    ax[1].set_xlabel('Arrival Rate (req/s)')
    ax[1].set_ylabel('End-to-End Latency (s)')
    ax[1].grid(axis='y', linestyle='--', alpha=0.4)

    handles = [
        Line2D([], [], color='#4d7cd1', marker='o', label='CPU Only'),
        Line2D([], [], color='#4c4c4c', marker='o', linestyle='-', label='4GPUs'),

        Line2D([], [], color='#76b900', marker='o', label='ALL-GPU'),
        Line2D([], [], color='#4c4c4c', marker='s', linestyle='--', label='6GPUs'),

        Line2D([], [], color='#FF912A', marker='o', label='vLiteRAG'),
        Line2D([], [], color='#4c4c4c', marker='^', linestyle='-.', label='8GPUs'),
    ]

    ax[1].legend(handles, [h.get_label() for h in handles], loc='upper right', fontsize=7.5, ncol=3, frameon=True)

    plt.tight_layout()
    plt.savefig(f'{str(PRJ_ROOT)}/figures/figure_17.pdf', dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser("Plotting util arguments")
    parser.add_argument("--f10", action='store_true')
    parser.add_argument("--f11", action='store_true')
    parser.add_argument("--f12", action='store_true')
    parser.add_argument("--f14", action='store_true')
    parser.add_argument("--f15", action='store_true')
    parser.add_argument("--f16", action='store_true')
    parser.add_argument("--f17", action='store_true')
    parser.add_argument("--main", action='store_true')
    parser.add_argument("--all", action='store_true')
    args = parser.parse_args()

    if args.all:
        plot_figure_10()
        plot_figure_11()
        plot_figure_12()
        plot_figure_14()
        plot_figure_15()
        plot_figure_16()
        plot_figure_17()
        return
    
    if args.main:
        plot_figure_10()
        plot_figure_11()
        plot_figure_12()
        return

    if args.f11:
        plot_figure_11()
        print("[VLITE] Figure 11 saved.")

    if args.f10:
        plot_figure_10()
        print("[VLITE] Figure 10 saved.")

    if args.f12:
        plot_figure_12()
        print("[VLITE] Figure 12 (TTFT breakdown) saved.")

    if args.f14:
        plot_figure_14()
        print("[VLITE] Figure 14 (Dispatcher ablation) saved.")

    if args.f15:
        plot_figure_15()
        print("[VLITE] Figure 15 (In/Out length ablation) saved.")

    if args.f16:
        plot_figure_16()
        print("[VLITE] Figure 16 (SLO ablation) saved.")

    if args.f17:
        plot_figure_17()
        print("[VLITE] Figure 17 (GPU count ablation) saved.")

if __name__ == "__main__":
    main()
