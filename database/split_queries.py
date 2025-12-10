import numpy as np
import argparse

from pathlib import Path
PRJ_ROOT = Path(__file__).resolve().parents[1]

def write_bin(file_path, data):
    with open(file_path, 'wb') as f:
        count, dim = data.shape
        np.array([count, dim], dtype=np.int32).tofile(f)
        data.tofile(f)
        
def read_fbin(file_path):
    with open(file_path, 'rb') as f:
        cnt, dim = np.fromfile(f, count=2, dtype=np.int32)
        data = np.fromfile(f, dtype=np.float32).reshape(cnt, dim)
    return data

# Function to split a numpy array into train and test sets with shuffled indices.
def split_with_index_tracking(data, test_ratio=0.2, shuffle=True):
    n_samples = data.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    test_size = int(n_samples * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    train_data = data[train_indices]
    test_data = data[test_indices]

    return train_data, test_data, train_indices, test_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset name")
    args = parser.parse_args()
    
    dataset_dir = str(PRJ_ROOT / 'database' / args.dataset)
    
    data = read_fbin(f"{dataset_dir}/queries.fbin" )
    train_ratio = 0.2 if data.shape[0] <= 10000 else 0.01
    
    train_data, test_data, train_indices, test_indices = split_with_index_tracking(data, 1 - train_ratio, True)
    write_bin(f"{dataset_dir}/train_qvec.fbin", train_data)
    write_bin(f"{dataset_dir}/test_qvec.fbin", test_data)
    write_bin(f"{dataset_dir}/train_qids.ibin", train_indices.reshape(-1, 1))
    write_bin(f"{dataset_dir}/test_qids.ibin", test_indices.reshape(-1, 1))

if __name__ == "__main__":
    main()
