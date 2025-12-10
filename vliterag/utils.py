import csv
import struct
import plyvel
import subprocess
import numpy as np
from pathlib import Path
from vliterag.configs import vLiteConfigs

def load_vectors(path, num=None):
    def read_bin(dtype):
        with open(path, 'rb') as f:
            cnt, dim = np.fromfile(f, count=2, dtype=np.int32)
            if num is not None:
                assert cnt >= num
            data = np.fromfile(f, dtype=dtype).reshape(cnt, dim)
        return data[:num, :]
    
    path = str(path)
    if path.endswith('.fbin'):
        return read_bin(np.float32)
    elif path.endswith('.ibin'):
        return read_bin(np.int64)
    elif path.endswith('.txt'):
        with open(path, 'r') as f:
            return f.read().split()
    else:
        raise ValueError("Unsupported vector file format.")

def mmap_load_vector(path, num=None):
    def read_bin(dtype):
        with open(path, 'rb') as f:
            header = f.read(8)
            cnt, dim = struct.unpack('ii', header)
        if num is not None:
            assert cnt >= num
        data = np.memmap(path, dtype=dtype, mode='r', offset=8, shape=(cnt, dim))
        return data[:num, :]

    path = str(path)
    dtype = np.float32 if path.endswith('.fbin') else np.int64 if path.endswith('.ibin') else None
    if dtype is not None:
        return read_bin(dtype)
    else:
        raise ValueError("Unsupported vector file format for memory mapping.")

def write_bin(file_path, data):
    with open(file_path, 'wb') as f:
        count, dim = data.shape
        np.array([count, dim], dtype=np.int32).tofile(f)
        data.tofile(f)

def get_gpu_memory():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,nounits,noheader"])
    output = output.decode("utf-8")
    mem_data = np.array([list(map(int, line.strip().split(", "))) for line in output.splitlines()])
    min_gpu = np.min(mem_data[:,0])
    max_usage = np.max(mem_data[:,1])
    min_free = np.min(mem_data[:,2])
    return min_gpu, max_usage, min_free

def save_mem_req(fp, vector_database, slo, mem_req):
    rows = None
    with open(fp, 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    
    done = False
    for row in rows:
        if row[0] == vector_database and row[1] == slo:
            row[2] = str(mem_req)
            done = True
    if not done:
        rows.append([vector_database, slo, str(mem_req)])
    
    with open(fp, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def random_prompt(input_len, batch_size=1):
    bags_of_words = ["one", "two", "six", "ten", "apple", 
                        "banana", "orange", "grape", "car", "bike",]
    prompts = []
    for _ in range(batch_size):
        random_ids = np.random.choice(len(bags_of_words), size=input_len)
        prompt = " ".join(np.array(bags_of_words)[random_ids])
        prompts.append(prompt)
    return prompts

def prepare_queries(vlite_cfg, qtype="test", need_texts=True):
    database_dir = Path(vlite_cfg.database_dir) / vlite_cfg.index
    vectors = load_vectors(database_dir / f"{qtype}_qvec.fbin")
    indices = load_vectors(database_dir / f"{qtype}_qids.ibin").reshape(-1)
    n_queries = len(indices)

    perm = np.random.permutation(n_queries)
    indices = indices[perm]
    vectors = vectors[perm, :]
    
    if need_texts:
        query_db_dir = str(vlite_cfg.database_dir / 'query_database')
        query_db = plyvel.DB(query_db_dir, create_if_missing=False)
        query_dict = {int(k): v for k, v in query_db}
        query_texts = [query_dict[qid].decode() for qid in indices]
        query_db.close()
    else:
        query_texts = None
    
    return vectors, query_texts