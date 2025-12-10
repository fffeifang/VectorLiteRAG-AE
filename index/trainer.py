import os
import time
import faiss
import struct
import argparse
import numpy as np

from pathlib import Path
from vliterag.utils import load_vectors, mmap_load_vector, write_bin
from faiss.contrib.ivf_tools import replace_ivf_quantizer

database_dir = Path(__file__).resolve().parent.parent / "database"

def cluster_on_gpu(vectors, save_path, nlist=0):
    ntotal, d = vectors.shape
    if nlist == 0:
        nlist = int(np.sqrt(ntotal))
    
    kmenas = faiss.Kmeans(d, nlist, niter=10, verbose=True, gpu=True)
    
    t1 = time.time()
    
    print(f"[Trainer] Training KMeans with {ntotal} samples on GPU...")
    kmenas.train(vectors)
    
    print(f"[Trainer] KMeans training completed in {time.time() - t1:.2f} seconds.")
    centroids = kmenas.centroids
    
    write_bin(f"{save_path}/centroids.fbin", centroids.astype(np.float32))
    
    return centroids

def build_ivf(vectors, save_dir, nlist=0, use_gpu=False, heterrag=False):
    file_name = "heterrag_ivfpq.index" if heterrag else 'ivfpq.index'
    save_path = f"{save_dir}/{file_name}"
    ntotal, d = vectors.shape
    if nlist == 0:
        nlist = int(np.sqrt(ntotal))
    
    M = int(d // 2)
    nbit = 4
    
    nsample = min(nlist * 39, ntotal)
    sample_idx = np.random.choice(ntotal, size=nsample, replace=False)
    train_data = vectors[sample_idx]
    
    if Path(f"{save_path}.tmp").exists():
        print(f"[Trainer] Loading existing IVF index from {save_path}.tmp")
        ivf = faiss.read_index(f"{save_path}.tmp")
        ivf.verbose = True
    else:
        t1 = time.time()
        
        if Path(f"{save_dir}/centroids.fbin").exists():
            centroids = load_vectors(f"{save_dir}/centroids.fbin")
        elif use_gpu:
            centroids = cluster_on_gpu(train_data, save_dir, nlist)

        quantizer = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_L2)
        quantizer.hnsw.efConstruction = 200
        quantizer.hnsw.efSearch = 500
        quantizer.verbose = True
        quantizer.add(centroids)
        
        ivf = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbit, faiss.METRIC_L2)
        ivf.by_residual = True
        ivf.verbose = True
        ivf.train(train_data)
            
        faiss.write_index(ivf, f"{save_path}.tmp")
        print(f"[Trainer] IVF index trained with {nsample} samples in {time.time() - t1:.2f} seconds.")
    
    t2 = time.time()
    
    ivf.add(vectors)
    print(f"[Trainer] {ivf.ntotal} vectors added to IVF index in {time.time() - t2:.2f} seconds.")
    
    faiss.write_index(ivf, save_path)
    print(f"[Trainer] Saved IVF index to {save_path}")
    
    if Path(f"{save_path}.tmp").exists():
        os.remove(f"{save_path}.tmp")
        
    return ivf
    
def build_fs_from_ivf(ivf, save_dir, heterrag=False):
    file_name = "heterrag_ivffs.index" if heterrag else 'ivffs.index'
    save_path = f"{save_dir}/{file_name}"
    ivfpq_fs = faiss.IndexIVFPQFastScan(ivf)
    faiss.write_index(ivfpq_fs, save_path)
    print(f"[VLITE] Saved IVF Fast Scan index to {save_path}")
    
    return ivfpq_fs

def find_groundtruth(vectors, queries, topk, save_dir):
    ntotal, d = vectors.shape
    flat_index = faiss.IndexFlatL2(d)
    flat_index.add(vectors)
    D, I = flat_index.search(queries, topk)
    
    write_bin(f"{save_dir}/groundtruth.{topk}.ibin", I.astype(np.int32))
    write_bin(f"{save_dir}/groundtruth.{topk}.fbin", D.astype(np.float32))
    print(f"[VLITE] Saved ground truth neighbors and distances to {save_dir}")
    
    return D, I

def load_database(vector_files):
    """ Loads large database files into a single numpy array """
    def get_header(path):
        with open(path, 'rb') as f:
            header = f.read(8)
            count, dim = struct.unpack('ii', header)
        return count, dim

    ntotal = 0
    dim = None
    for f in vector_files:
        cnt, d = get_header(f)
        ntotal += cnt
        if dim is None:
            dim = d
        elif dim != d:
            raise ValueError("Inconsistent vector dimensions in database files.")
    
    merged = np.empty((ntotal, dim), dtype=np.float32)
    
    offset = 0
    for f in vector_files:
        arr = mmap_load_vector(f)
        count = arr.shape[0]
        merged[offset:offset+count] = arr[:]
        offset += count
    
    return merged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_list", "-n", type=int, default=0, help="Number of IVF lists in Ki (0 to auto-compute)")
    parser.add_argument("--topk", "-k", type=int, default=100, help="Number of nearest neighbors for ground truth")
    parser.add_argument("--dataset", "-d", help="Dataset type")
    parser.add_argument("--operation", "-op", choices=["build_ivf", "build_fs", "find_gt"], required=True, help="Operation to perform")
    parser.add_argument("--use_gpu", '-g', action="store_true", help="Use GPU for training IVF index")
    parser.add_argument("--heterrag", '-H', action='store_true', help="Build Index for HeterRAG")
    args = parser.parse_args()
    
    vector_files = list((Path(database_dir) / args.dataset).glob('base*.fbin'))
    query_file = Path(database_dir) / args.dataset / 'queries.fbin'
    output_dir = Path(database_dir) / args.dataset
    
    if not vector_files or not query_file.exists():
        raise FileNotFoundError("Vector database files not found.")
    
    if args.operation == 'build_ivf':
        vectors = load_database(vector_files)
        nlist = args.num_list * 1024 if args.num_list > 0 else 0
        ivf = build_ivf(vectors, output_dir, nlist, args.use_gpu, args.heterrag)
        
    elif args.operation == 'build_fs':
        index_file = "heterrag_ivfpq.index" if args.heterrag else "ivfpq.index"
        ivf_path = f"{output_dir}/{index_file}"
        if not Path(ivf_path).exists():
            print("[Trainer] IVF index file not found. Building IVF index first...")
            vectors = load_database(vector_files)
            nlist = args.num_list * 1024 if args.num_list > 0 else 0
            ivf = build_ivf(vectors, output_dir, nlist, args.use_gpu, args.heterrag)
        else:
            ivf = faiss.read_index(ivf_path)
        
        print("[Trainer] Building IVF Fast Scan index from IVF index...")
        ivffs = build_fs_from_ivf(ivf, output_dir, args.heterrag)
        
    elif args.operation == 'find_gt':
        vectors = load_database(vector_files)
        queries = load_vectors(query_file)
        find_groundtruth(vectors, queries, args.topk, output_dir)
        
if __name__ == "__main__":
    main()