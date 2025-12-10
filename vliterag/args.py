import argparse

def parse_args():
    p = argparse.ArgumentParser(description="VectorLite RAG Launcher")

    p.add_argument("--model", type=str, default='llama8b')
    p.add_argument("--index", type=str, default='wikiall')

    p.add_argument("--gpu_type", type=str, default="L40S", choices=["L40S", "H100"])
    p.add_argument("--num_gpus", type=int, default=8)
    p.add_argument("--gpu_util", type=float, default=0.95)

    p.add_argument("--tp_size", type=int, default=0)
    p.add_argument("--input_len", type=int, default=1024)
    p.add_argument("--output_len", type=int, default=256)

    p.add_argument("--search_mode", type=str, default=None,
                    choices=["all", "cpu", "all-gpu", "ded-gpu", "vlite", "hedrarag"])
    p.add_argument("--search_slo", type=int, default=-1)
    p.add_argument("--search_topk", type=int, default=25)
    p.add_argument("--search_nprobe", type=int, default=2048)
    p.add_argument("--ann_workers", type=int, default=1)

    p.add_argument("--disable_dispatcher", action="store_true")
    p.add_argument("--sweep", action='store_true')
    p.add_argument("--is_profiling", action="store_true")

    p.add_argument("--arrival_rate", type=float, default=0.0)
    p.add_argument("--running_time", type=int, default=120)

    p.add_argument("--result_dir", type=str, default="results")
    p.add_argument("--file_name", type=str, default="")
    p.add_argument("--tag", type=str, default="main")

    return p.parse_args()