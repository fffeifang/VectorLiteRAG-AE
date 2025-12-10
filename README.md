### VectorLiteRAG Artifact README
This repository contains the artifact for VectorLiteRAG, including the full implementation, experiment scripts, preprocessing pipelines, and plotting utilities required to reproduce the evaluation results reported in the paper.
The artifact supports both end-to-end evaluation runs and fine-grained, individual executions for profiling and ablation studies.

## Environment Setup
#### Setup conda environment
The artifact is tested using Anaconda / Miniconda.
A complete environment specification is provided.

`conda env create -f scripts/env.yml
conda activate vlite`

#### Build native dependencies.
After activating the environment build the required native component.
`git submodule update --init
./scripts/build.sh
`

This step compiles modified FAISS (from version 1.9.0) with GPU support and links required libraries including Intel MKL.

## Model Cache Configuraiton
HuggingFace models are downloaded automatically at runtime.
You must manually specify a local cache directory to avoid repeated downloads.

Edit the following file:
`vliterag/engine.py`

at the top of the file, set 
`model_cache = "/path/to/your/model/cache"`

Ensure the directory has sufficient disk space (models up to 70B parameters)

## Dataset Preparation and Test Runs
#### Dataset Download (1 ~ 2 hrs)
Download a dataset using:
`./database/download.sh <dataset>`
Supported datasets are:
* wikiall
* orcas1k
* orcas2k

#### Encoding and Index Construction 
For datasets requiring preprocessing (e.g., ORCAS benchmarks):
`./database/encode.sh <dataset>`
`./scripts/train.sh <dataset>`
This step performs:
* Document chunking
* Vector encoding
* IVF index training

Note, full preprocessing can take 40 ~ 60 hrs and requires 1.5TB ~ 2TB storage & system memory

#### Test Run (Sanity Check)
A lightweight test run can be executed before full evaluation (discarded immediately after execution):
`./scripts/runall_l40s.sh test`
This performs a small CPU-based run on Wiki-All to verify correctness.

#### Full Evaluation Runs
##### L40S Node:
`./scripts/runall_l40s.sh <main|inout|dispatcher>`
##### H100 Node:
`./scripts/runall_h100.sh <main|inout|slo|ngpu>`
Available options:
* main - evaluation sweep for figure 10 and 11
* inout - input / output length ablation
* dispatcher - dispatcher on/off ablation
* slo - slo level ablation
* ngpu - gpu number ablation

## Results and Plotting
#### Result Files
Evaluation outputs are soted under:
`VectorLiteRAG/results/`
Structure:
`results/<index>/<model>/<ngpus>gpus/<mode>/
  ├── raw/        # request-level parquet files
  ├── summary/    # aggregated CSV summaries`

#### Plotting Figures
All plotting scripts are centralized under 
`analysis//plot.py`

Use the provided wrapper sript after running evaluations:
`./scripts/plotall.sh <args>`

Examples:
`./scripts/plotall.sh all     # generate all figures
./scripts/plotall.sh main    # generate main figures only
./scripts/plotall.sh 14      # generate Figure 14 only`

Generated figures are saved to:
`VectorLiteRAG/figures/`

## Individual Runs via main.py
Advanced users can invoke runs directly

#### Profiling Mode
Used to generate latency models and partitioned indexes (is_profiling = True)
`python main.py \
  --model llama8b \
  --index wikiall \
  --is_profiling`

#### Serving / Evalation Mode
Example single run:
`python main.py \
  --model llama8b \
  --index orcas2k \
  --search_mode vlite \
  --arrival_rate 32 \
  --input_len 1024 \
  --output_len 256`
Sweep mode:
  `python main.py \
  --model llama8b \
  --index orcas2k \
  --search_mode all \
  --sweep`

Notes
* Results may vary slightly depending on hardware conditions.
* Large-scale preprocessing is expected to be time- and storage-intensive.
* All scripts are designed to run from the project root directory.
* configuration flexibility are currently limited; some parameters must be modified directly through the JSON files in `VectorLiteRAG/configs`.
