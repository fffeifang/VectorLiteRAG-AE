import os
import uuid
import plyvel
import torch
from argparse import ArgumentParser
import numpy as np
import multiprocessing as mp
from multiprocessing import Queue
from multiprocessing import set_start_method
from sentence_transformers import SentenceTransformer

from pathlib import Path
PRJ_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "dunzhang/stella_en_1.5B_v5"
CACHE_DIR = str(PRJ_ROOT / 'database' / "embedding_model")
BATCH_SIZE = 128
TMP_DIR =  str(PRJ_ROOT / 'database' / "tmp_embeds")

def worker_process(gpu_id, texts_subset, tmp_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Loading model on GPU {gpu_id}...")
    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=CACHE_DIR,
        trust_remote_code=True
    ).cuda()
    print(f"[GPU {gpu_id}] Encoding {len(texts_subset)} texts...")

    show_progress = True if gpu_id == torch.cuda.device_count() - 1 else False
    embeddings = model.encode(
        texts_subset,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        show_progress_bar=show_progress,
    ).cpu().numpy()

    np.save(tmp_path, embeddings)
    print(f"[GPU {gpu_id}] Saved embeddings to {tmp_path}")

def encode_to_embeddings_mp(texts, num_gpu):
    total = len(texts)
    chunk = total // num_gpu

    os.makedirs(TMP_DIR, exist_ok=True)

    processes = []
    tmp_files = []
    
    for gpu_id in range(num_gpu):
        start = gpu_id * chunk
        end = total if gpu_id == num_gpu - 1 else (gpu_id + 1) * chunk
        subset = texts[start:end]

        tmp_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}.npy")
        tmp_files.append(tmp_path)

        p = mp.Process(
            target=worker_process,
            args=(gpu_id, subset, tmp_path)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Merging temporary embedding files...")
    arrays = [np.load(path) for path in tmp_files]
    final_embeddings = np.concatenate(arrays, axis=0)

    for path in tmp_files:
        os.remove(path)

    return final_embeddings

def write_fbin(path, embeddings):
    outfile = f"{path}.fbin"
    with open(outfile, "wb") as f:
        f.write(np.array(embeddings.shape, dtype=np.int32).tobytes())
        f.write(embeddings.tobytes())
    print(f"Saved embeddings: {embeddings.shape} → {outfile}")

def read_from_db(path):
    db = plyvel.DB(path, create_if_missing=False)
    texts = [v.decode("utf-8") for _, v in db]
    db.close()
    print(f"Loaded {len(texts)} items from {path}")
    return texts

def main():
    parser = ArgumentParser()
    parser.add_argument("--path", "-p", help="Path to LevelDB")
    parser.add_argument("--output", "-o", help="Output file prefix")
    parser.add_argument("--shard", "-s", type=int, default=0, help="Number of shards to split the database into")
    parser.add_argument("--init", action="store_true", help="Initialize the embedding model cache")
    args = parser.parse_args()
    
    if args.init:
        _ = SentenceTransformer(
            MODEL_NAME,
            cache_folder=CACHE_DIR,
            trust_remote_code=True
        )
        print("Model cache initialized.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available.")

    num_gpu = torch.cuda.device_count()
    print(f"Using {num_gpu} GPUs")

    texts = read_from_db(args.path)
    
    # if database is too large, it is saved into multiple files
    chunk_size = 2 ** 24    # 4 million ~ 32GB for float32, 2048-dim
    if len(texts) > chunk_size and 'base' in args.output:
        
        print(f"Database too large ({len(texts)} items). Splitting into chunks...")
        num_iter = (len(texts) + chunk_size - 1) // chunk_size
        
        for i in range(0, num_iter):
            chunk_output = f"{args.output}_{i}"
            if os.path.exists(f"{chunk_output}.fbin"):
                print(f"Chunk {i} already exists. Skipping...")
                continue
            
            print(f"Processing chunk {i} / {num_iter - 1}...")
            chunk_texts = texts[i * chunk_size:(i + 1) * chunk_size]
            embeddings = encode_to_embeddings_mp(chunk_texts, num_gpu)
            write_fbin(chunk_output, embeddings)
    else:        
        embeddings = encode_to_embeddings_mp(texts, num_gpu)
        write_fbin(args.output, embeddings)
    
    print("Encoding Complete.")
    
if __name__ == "__main__":
    set_start_method('spawn', force=True)
    main()