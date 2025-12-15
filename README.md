# Janus

A Dual-Mask Attention Transformer for Log-based Anomaly Detection in Cellular Networks

## Step-by-Step Training Workflow

The four training phases can be executed sequentially once the `data/` directory contains the logs, spec files and code. Janus training now consists of two stages. The first stage adapts the base model to 5G concepts, and the second stage fine-tunes the model on log data.


**Clone and install dependencies**:

   ```bash
   git clone --recurse-submodules  https://github.com/UmakantKulkarni/janus.git
   cd janus
   ```

## Installation instructions for x86_64 system:
```bash
pip3 install -U accelerate datasets fastapi fitz Flask ijson omegaconf pandas peft pydantic PyGithub pytest PyYAML rank_bm25 scikit_learn torch tqdm transformers uvicorn tokenizers tensorflow tf-keras transformers[torch] sentencepiece sentence-transformers faiss-cpu trl rouge_score fire deepspeed matplotlib wandb gradio download_3gpp fairscale tiktoken blobfile python-docx pypdf2 pymupdf pypandoc
```

## Manual Setup on IBM Power Systems

The project can run on a bare-metal installation with two Tesla P100 GPUs and CUDA 11.6. The steps below assume a Conda environment with Python 3.10:

# General installation instructions for IBM Power Systems
```bash
# Make sure torch is built from source in py310_env
conda create --name devenv --clone py310_env
conda activate devenv

# install torch 2.6.0 from source
python -c "import torch; print(torch.cuda.is_available())"

# make sure cmake points to conda cmake and not to /usr/
conda install -y -c conda-forge cmake=3.29
conda install -y -c conda-forge pyarrow
pip install transformers peft datasets accelerate
pip install typing_extensions==4.14.0
sudo apt install -y libxml2-dev libxslt1-dev
sudo apt-get install -y golang
conda install -y -c conda-forge wandb
conda install -y -c conda-forge pillow
pip install PyGithub tokenizers flask sentencepiece sentence-transformers trl safetensors scipy regex pandas rouge_score fire deepspeed matplotlib wandb gradio download_3gpp fairscale tiktoken blobfile python-docx tqdm pypdf2 pypandoc

sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt -y install gcc-11 g++-11
pip install pymongo
pip install --force-reinstall -v "numpy==1.25.2" scipy
pip install ijson
pip install transformers==4.47.1 # https://github.com/unslothai/unsloth/issues/1476

# install bitsandbytes
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git && cd bitsandbytes/
cmake -DCOMPUTE_BACKEND=cuda -S .
make
pip install -e .

# install faiss
git clone https://github.com/facebookresearch/faiss.git
cd faiss
git checkout v1.11.0
apt-get -y install swig
cmake . -B build -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DFAISS_OPT_LEVEL=generic -DCMAKE_CUDA_ARCHITECTURES="60"
cmake --build build --config Release -j
cmake --install build
cd build/faiss/python  
python3 setup.py bdist_wheel
cd ../../../
ls build/faiss/python/dist/faiss-1.11.0-py3-none-any.whl
pip install rank-bm25
pip install hydra-core

conda install -y -c conda-forge libstdcxx-ng libgcc-ng
conda install -y -c conda-forge "libblas=*=*openblas" libopenblas
echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"' > "$CONDA_PREFIX/etc/conda/activate.d/00-prepend-lib.sh"

python -c "import pyarrow, transformers; print('OK')"

find /usr -name libffi.so.7
ln -sf  /usr/lib/powerpc64le-linux-gnu/libffi.so.7 /opt/anaconda3/envs/devenv/lib/libffi.so.7

PYTHONPATH=$(pwd)/janus python3 janus/janus/preprocess/build_pretraining_corpus.py
```

#  Accelerate config (No FSDP; Only DDP)
```bash
(plelog) root@node0:/opt/janus# cat /users/umakant/.cache/huggingface/accelerate/default_config.yaml
compute_environment: LOCAL_MACHINE
debug: true
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
(plelog) root@node0:/opt/janus#
```

# run style checks and tests
```bash
ruff janus
PYTHONPATH=$PWD/janus pytest -q
```

## Running inference on a log file

The CLI script `janus/janus/infer/inference.py` scores a single network function log and appends the detailed results to a CSV file. The script loads the base LLaMA model together with the PEFT adapter and classification head, formats the log exactly as during training (including the `NF:<name>` prefix on every window) and strips ANSI escape sequences before scoring.

1. Ensure `config/default.yaml` (or your repo configuration) points to the trained base model directory and the final adapter directory.

2. Run the inference script from the repository root:

   ```bash
   PYTHONPATH=$(pwd)/janus python3 janus/janus/infer/inference.py data/raw_data/logs/UE-INITIATED-DEREGISTRATION-PROCEDURE/open5gs-amf-deployment-5b4c845869-n74bh-amf.log --network-function amf  --csv-log-file /tmp/inference_results.csv --calibrate-dir artifacts/calibrate --max-tokens 1024
   
   PYTHONPATH=$(pwd)/janus python3 janus/janus/infer/inference.py data/raw_data/anomalous_logs/amf/batch_001_amf/amf_open5gs-amf-deployment-5b4c845869-n74bh-amf.log --network-function amf  --csv-log-file /tmp/inference_results.csv --calibrate-dir artifacts/calibrate --max-tokens 1024
   ```

   The network function is inferred from the filename or the log contents. Override it with `--network-function AMF` when necessary. Use `--max-tokens` to match a different training sequence length (defaults to `train.seq_len`). Omit `--calibrate-dir` to skip calibration statistics (the related CSV columns will be filled with `NaN`).

3. The script prints a summary to the console and appends a CSV row with the timestamp, input filename, inferred network function and a wide range of classification-head statistics: probability and logit aggregates (mean, variance, quartiles), entropy, and the ratio of windows exceeding 0.5/0.8/0.9 probability thresholds. When `--calibrate-dir` is supplied the CSV also includes percentile/z-score/MAD anomaly metrics against the provided calibration set; otherwise those columns are `NaN`.

# Prepare dataset
```bash
PYTHONPATH=$(pwd)/janus python3 janus/janus/preprocess/control_flow_graph.py

PYTHONPATH=$(pwd)/janus python3 janus/janus/preprocess/bugs/scrapeOpen5gsIssues.py
PYTHONPATH=$(pwd)/janus python3 janus/janus/preprocess/bugs/scrapeOpen5gsCommits
PYTHONPATH=$(pwd)/janus python3 janus/janus/preprocess/bugs/merge_jsons.py
PYTHONPATH=$(pwd)/janus python3 janus/janus/preprocess/bugs/filter_bugs.py /opt/janus/data/preprocessed_data/mapped_issues.json /opt/janus/data/preprocessed_data/open5gs_bugs.json
PYTHONPATH=$(pwd)/janus python3 janus/janus/preprocess/bugs/prepare_bug_data.py
PYTHONPATH=$(pwd)/janus python3 janus/janus/preprocess/bugs/prepare_defect_data.py
PYTHONPATH=$(pwd)/janus python3 janus/janus/preprocess/bugs/get_eval_defects.py

bash janus/janus/preprocess/specs/get_specs.sh
PYTHONPATH=$(pwd)/janus python3 janus/janus/preprocess/specs/processtspdf.py
PYTHONPATH=$(pwd)/janus python3 janus/janus/preprocess/specs/filter_specs.py
rm /opt/janus/data/raw_data/spec_3gpp/ts_3gpp_dataset.json
bash janus/janus/preprocess/specs/get_spec_yml.sh
bash janus/janus/preprocess/specs/filter_spec_yaml.sh

PYTHONPATH=$(pwd)/janus python3 janus/janus/preprocess/build_pretraining_corpus.py

find /opt/janus/data/raw_data/logs -type f -name "*.log" -exec head -n 2000 {} \; > /opt/janus/data/preprocessed_data/clean_logs_for_pairing.txt

nohup bash -c 'PYTHONPATH=$(pwd)/janus python3 janus/janus/preprocess/build_explainability_indexes.py --openapi-dir /opt/janus/data/raw_data/3gpp_openapi_yaml_rel17 --source-dir /opt/janus/data/raw_data/open5gs_source_code --spec-path /opt/janus/data/raw_data/spec_3gpp/filtered_3gpp_specs.json --output-dir /opt/janus/data/preprocessed_data' &
```

# Run training using curriculum learning
```bash
nohup bash -c 'PYTHONPATH=$(pwd)/janus accelerate launch --num_processes=2 --num_machines=1 --machine_rank=0 -m janus.train.cpt --config janus/janus/config/cpt.yml' &

nohup bash -c 'PYTHONPATH=$(pwd)/janus accelerate launch --num_processes=2 --num_machines=1 --machine_rank=0 -m janus.train.train --config janus/janus/config/warmup.yml' &
torchrun --nproc_per_node=2 janus/janus/train/dtensor_to_tensor.py --dtensor_dir artifacts/warmup_adapter_final

nohup bash -c 'PYTHONPATH=$(pwd)/janus accelerate launch --num_processes=2 --num_machines=1 --machine_rank=0 -m janus.train.train --config janus/janus/config/dualpass.yml' &
torchrun --nproc_per_node=2 janus/janus/train/dtensor_to_tensor.py --dtensor_dir artifacts/dualpass_adapter_final

nohup bash -c 'PYTHONPATH=$(pwd)/janus accelerate launch --num_processes=2 --num_machines=1 --machine_rank=0 -m janus.train.train_defects --config janus/janus/config/train_defects.yml' &
torchrun --nproc_per_node=2 janus/janus/train/dtensor_to_tensor.py --dtensor_dir artifacts/defects_adapter_final
python3 janus/janus/scripts/calculate_thresholds.py --log-file artifacts/logs/defects.log --output-file artifacts/thresholds.json

nohup bash -c 'PYTHONPATH=$(pwd)/janus accelerate launch --num_processes=2 --num_machines=1 --machine_rank=0 -m janus.train.build_nf_reference --config janus/janus/config/calibrate.yml' &
```

# Run bulk evaluation for sigmetrics26
```bash
nohup bash -c 'PYTHONPATH=$(pwd)/janus python3 janus/janus/scripts/run_open5gs_evaluation.py' &
PYTHONPATH=$(pwd)/janus python3 janus/janus/scripts/generate_eval_metadata.py
PYTHONPATH=$(pwd)/janus python3 janus/janus/scripts/get_pred_true_labels.py --csv_path plelog.csv
PYTHONPATH=$(pwd)/janus python3 janus/janus/scripts/get_pred_true_labels.py --csv_path neurallog.csv --predict_labels 0
PYTHONPATH=$(pwd)/janus python3 janus/janus/scripts/get_pred_true_labels.py --csv_path janus.csv --nf_thresholds artifacts/thresholds.json
python3 janus/janus/scripts/compute_accuracy.py --csv_path janus.csv
python3 janus/janus/scripts/compute_accuracy.py --csv_path plelog.csv
python3 janus/janus/scripts/compute_accuracy.py --csv_path neurallog.csv
```

## Data Format

All datasets live under the top-level `data/` directory.

- Place training data from https://huggingface.co/datasets/umakantk/janusdata under `data/raw_data`.
- Place evaluation data under `data/eval_data`.

```
data/
├── raw_data/
│   ├── logs/
│   │   └── <call_flow>/<sessions>/<nf>/*.log
│   ├── spec_3gpp/
│   │   └── *.json              # objects with a `content` field
│   └── open5gs_source_code/
├── eval_data/
    └── logs/
        └── <call_flow>/<sessions>/<nf>/*.log
```
   