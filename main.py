import asyncio

from configs.loader import load_all_models, load_index

from vliterag.args import parse_args
from vliterag.configs import vLiteConfigs
from vliterag.profiler import profile
from vliterag.extractor import extract
from vliterag.runner import run_pipeline

ALL_INDEXES = list(load_index().keys())
ALL_MODELS = list(load_all_models().keys())
ALL_MODES = ['cpu', 'all-gpu', 'ded-gpu', 'vlite']

def main():
    args = parse_args()

    # -------------------------------
    # PROFILING / PARTITIONING
    # -------------------------------
    if args.is_profiling:
        if args.model == 'all' and args.index == 'all':
            profile_all(args)
        elif args.model == 'all':
            profile_models(args)
        elif args.index == 'all':
            profile_indexes(args)
        else:
            profile_single(args, args.sweep)
        return

    # -------------------------------
    # NORMAL RUN 
    # -------------------------------
    if args.model == 'all' and args.index == 'all':
        sweep_all(args)
    elif args.model == 'all':
        sweep_models(args)
    elif args.index == 'all':
        sweep_indexes(args)
    elif args.sweep:
        sweep_single(args)
    else:
        run_single(args)
        
def sweep_all(args):
    if args.search_mode == 'all':
        modes = ALL_MODES
    else:
        modes = [args.search_mode]
    
    for mode in modes:
        args.search_mode = mode
        for index in ALL_INDEXES:
            args.index = index
            sweep_models(args)

def sweep_models(args):
    if args.search_mode == 'all':
        modes = ALL_MODES
    else:
        modes = [args.search_mode]
    
    for mode in modes:
        args.search_mode = mode
        for model in ALL_MODELS:
            args.model = model
            sweep_single(args)

def sweep_indexes(args):
    if args.search_mode == 'all':
        modes = ALL_MODES
    else:
        modes = [args.search_mode]
    for mode in modes:
        args.search_mode = mode
        for index in ALL_INDEXES:
            args.index = index
            sweep_single(args)

def sweep_single(args):
    if args.search_mode == 'all':
        modes = ALL_MODES
    else:
        modes = [args.search_mode]
    
    for mode in modes:
        args.search_mode = mode
        config = build_run_config(args, sweep=True)
        asyncio.run(run_pipeline(config))
    
def run_single(args):
    if args.search_mode == 'all':
        modes = ALL_MODES
    else:
        modes = [args.search_mode]
    
    for mode in modes:
        args.search_mode = mode
        config = build_run_config(args, sweep=False)
        asyncio.run(run_pipeline(config))

# ----- Profiler launchers -----
def profile_all(args):
    for index in ALL_INDEXES:
        args.index = index
        profile_models(args)

def profile_models(args):
    for model in ALL_MODELS:
        args.model = model
        profile_single(args, True)
        
def profile_indexes(args):
    for index in ALL_INDEXES:
        args.index = index
        profile_single(args, True)
        
def profile_single(args, sweep=False):
    config = build_profile_config(args, sweep)
    profile(config)
    extract(config)

def build_run_config(args, sweep=False):
    return vLiteConfigs(
        model=args.model,
        index=args.index,
        num_gpus=args.num_gpus,
        input_len=args.input_len,
        output_len=args.output_len,
        search_mode=args.search_mode,
        search_slo=args.search_slo,
        search_nprobe=args.search_nprobe,
        ann_workers=args.ann_workers,
        dispatcher=not args.disable_dispatcher,
        sweep=sweep,
        is_profiling=False,
        arrival_rate=args.arrival_rate,
        running_time=args.running_time,
        file_tag=args.tag
    )

def build_profile_config(args, sweep=False):
    # We infer gpu from model as we used the same combinations in the artifacts
    gpu_type = "L40S" if args.model == "llama8b" else "H100"
    
    return vLiteConfigs(
        model=args.model,
        index=args.index,
        gpu_type=gpu_type,
        num_gpus=args.num_gpus,
        input_len=args.input_len,
        output_len=args.output_len,
        search_slo=args.search_slo,
        search_mode='cpu',          # profiling always uses CPU mode
        search_nprobe=args.search_nprobe,
        dispatcher=False,
        sweep=sweep,
        is_profiling=True,
        arrival_rate=0,
        running_time=args.running_time,
        file_tag=args.tag
    )
    
if __name__ == "__main__":
    main()