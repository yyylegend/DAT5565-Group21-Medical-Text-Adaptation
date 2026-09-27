# DAT5565 Group 21 — Medical Text Adaptation

**Languages:** English | [简体中文](README.zh-CN.md)

This repository contains the data preparation and TensorFlow/Keras continued-pretraining (CPT) pilot for our medical question-answering project. The [research overview](docs/research/README.md) explains the current Base → CPT → SFT plan and datasets; its [English version](docs/research/README.en.md) is available for comparison. The instructions below target a Linux GPU server, such as a rented cloud instance. Local Windows users do not need to start WSL to transfer or inspect the prepared data.

## Project layout

```text
data/raw/                 downloaded source archives (not tracked by Git)
data/processed/           cleaned splits and manifests (not tracked by Git)
docs/research/            bilingual research overview and execution plan
models/                   local model cache and presets (not tracked by Git)
runs/                     adapters and run metadata (not tracked by Git)
src/healthcpt/            data audit, preparation, and CPT commands
pyproject.toml            project dependencies and Python version
uv.lock                   resolved dependency versions
```

The Python package keeps only the scripts needed by this workflow:

| File | Purpose |
|---|---|
| `src/healthcpt/medquad.py` | Audit MedQuAD and create source-disjoint QA splits. |
| `src/healthcpt/medical_data.py` | Download and clean MedlinePlus/PMC text, sample CPT sources, and create text chunks and evaluation files. |
| `src/healthcpt/cpt.py` | Run a bounded TensorFlow/KerasHub CPT pilot. |
| `src/healthcpt/cli.py` | Expose the project commands through `healthcpt`. |

## Server requirements

- Linux x86_64 (Ubuntu is a convenient choice), Python 3.11 or 3.12, and `uv`.
- An NVIDIA GPU with a compatible driver installed by the server administrator. Confirm that `nvidia-smi` runs before starting.
- Internet access on the first setup to download Python packages and model weights from the selected preset source, unless those have been cached or mirrored on the server.

The lock file currently resolves Python 3.11/3.12, TensorFlow 2.21, Keras 3.15, and KerasHub 0.32. Use a clean Linux GPU image and let `uv sync --locked` install this environment. Python 3.8 and TensorFlow 1.x environments are incompatible with this project. The server still needs a compatible NVIDIA driver. See the [TensorFlow installation guide](https://www.tensorflow.org/install/pip).

KerasHub downloads the model preset on its first load. Point `KAGGLEHUB_CACHE` to persistent server storage before training so the model does not need to be downloaded again after an instance restart:

```bash
mkdir -p /path/to/persistent-storage/kagglehub
export KAGGLEHUB_CACHE=/path/to/persistent-storage/kagglehub
```

You can also use the official Hugging Face checkpoint. To download it into the ignored `models/` folder before training, set the Hugging Face cache and run:

```bash
mkdir -p models/huggingface
export HF_HOME="$PWD/models/huggingface"
uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-2B-Base'))"
```

Then use `--preset hf://Qwen/Qwen3.5-2B-Base`. KerasHub converts compatible Hugging Face Safetensors checkpoints when loading them; verify this specific model with a small pilot before the full run.

Install `uv` using the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

## Set up the project

Clone the repository into a persistent directory on your Linux GPU server. Replace the placeholders with your persistent directory and GitHub repository URL:

```bash
cd YOUR_PERSISTENT_DIRECTORY
git clone YOUR_GROUP_REPOSITORY_URL DAT5565-Final-Project
cd DAT5565-Final-Project
uv sync --locked
```

`uv sync --locked` creates the project environment and installs the versions recorded in `uv.lock`. Then confirm that TensorFlow can see the GPU:

```bash
nvidia-smi
uv run python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

The second command should print at least one GPU. If it prints an empty list, resolve the server's driver or GPU access configuration before launching CPT.

## Prepare MedQuAD

The raw archive and generated data are excluded from Git. Download the [original MedQuAD repository](https://github.com/abachaa/MedQuAD) archive on the server, or transfer the same ZIP file to the server, and place it at `data/raw/medquad-master.zip`:

```bash
mkdir -p data/raw
curl -L https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip \
  -o data/raw/medquad-master.zip
```

The counts in the [research overview](docs/research/README.md) came from an archive with SHA-256 `45aeef400844f3551a7862c3378cc9edf72818ef09d6c1d400f227207ee5179d`. The `master` archive may change. Check a newly downloaded file before relying on those counts:

```bash
sha256sum data/raw/medquad-master.zip
```

Run the audit and create the prepared question-answer and CPT splits:

```bash
uv run healthcpt audit-medquad data/raw/medquad-master.zip
uv run healthcpt prepare-medquad \
  data/raw/medquad-master.zip \
  data/processed/medquad-v1
```

Preparation removes records with missing questions or answers and exact duplicate question-answer pairs. It groups records by source URL before making train, validation, and test splits. CPT text is generated from the training and validation splits only; test content is kept out of CPT. The manifest records the archive SHA-256 and split counts. The audited archive yields 16,359 unique complete QA pairs from 5,486 XML files that contain at least one complete pair. For v3 CPT, each text chunk is one training example: there are 24,240 chunks from 4,401 source-document IDs. The [research overview](docs/research/README.md) shows the source breakdown and word-count mix.

## Add medical text for CPT

Download the latest MedlinePlus Health Topic XML and a bounded PMC OA sample selected from common MedQuAD training topics. The default requests up to 30 English CC0/CC BY articles per topic across 20 topics (up to 600 articles); it does not download the whole PMC collection.

```bash
uv run healthcpt download-medical-sources \
  data/processed/medquad-v1/qa_train.jsonl \
  data/raw/medical-sources-v3 \
  --topic-limit 20 --articles-per-topic 30
```

The source files are saved under `data/raw/medical-sources-v3/medlineplus/` and `data/raw/medical-sources-v3/pmc_oa/`. Using a versioned folder keeps earlier samples intact.

Clean and combine the CPT text. This keeps MedQuAD QA files separate, excludes MedlinePlus pages tied to held-out questions, and creates evaluation files with exact train-answer overlaps removed:

```bash
uv run healthcpt prepare-medical-corpus \
  data/processed/medquad-v1 \
  data/raw/medical-sources-v3 \
  data/processed/cpt-medical-v3 \
  --medquad-cpt-limit 6000 --chunk-words 200
```

The preparation keeps 6,000 sampled MedQuAD answer texts for CPT, leaves the full MedQuAD SFT training split intact, strips embedded HTML markup from MedlinePlus summaries, and splits long source text into chunks of at most 200 whitespace-separated words. The manifest records chunk and source-document counts, licenses, hashes and overlap removals. The current v3 output contains 24,240 CPT training chunks, 1,645 CPT validation chunks, 1,471 cleaned QA validation examples and 1,573 cleaned QA test examples. Keep the manifest and these files with the raw data when moving to the training server; `data/` is excluded from Git.

## Run CPT

Start with a small Qwen3.5-2B pilot on the v3 chunks to check that the preset loads, training runs, and the LoRA adapter saves:

```bash
uv run healthcpt cpt-pilot \
  data/processed/cpt-medical-v3/cpt_train.jsonl \
  data/processed/cpt-medical-v3/cpt_validation.jsonl \
  runs/qwen3_5_2b_cpt_pilot \
  --preset qwen3_5_2b_base \
  --limit-train 32 --limit-validation 8 \
  --sequence-length 512 --batch-size 1 --epochs 1 --lora-rank 8
```

The script prints the number of examples and steps before loading the model. During training, Keras shows the epoch and step progress, estimated time remaining (ETA), and training loss; validation loss appears after each validation pass. The ETA is based on observed batch speed and may change during the run. These losses measure the language-modeling objective, not answer quality.

The script also writes TensorBoard scalars every 100 training steps under `<output_dir>/tensorboard`, including training loss, learning rate, and steps per second; validation loss is written after validation. The terminal progress bar remains the place to see ETA. In a second server terminal, start TensorBoard for the full run with:

```bash
uv run tensorboard \
  --logdir runs/qwen3_5_2b_cpt_full_1epoch/tensorboard \
  --host 127.0.0.1 --port 6006
```

Open the dashboard through an SSH tunnel or the server provider's port-forwarding feature.

KerasHub 0.32 includes the `qwen3_5_2b_base` preset and Qwen3.5 model classes, but this project's path has not yet been run successfully on the target server. Treat this as a pilot only. Check the model load, LoRA target layers, memory use, and saved adapter before increasing the sample limits. The SFT training command is not implemented yet.

After the pilot succeeds, run one full pass over the current v3 dataset (24,240 training chunks and 1,645 validation chunks). This uses a 5% linear warmup, cosine decay to 10% of the peak learning rate, and a checkpoint every 2,000 steps:

```bash
uv run healthcpt cpt-pilot \
  data/processed/cpt-medical-v3/cpt_train.jsonl \
  data/processed/cpt-medical-v3/cpt_validation.jsonl \
  runs/qwen3_5_2b_cpt_full_1epoch \
  --preset qwen3_5_2b_base \
  --limit-train 24240 --limit-validation 1645 \
  --sequence-length 512 --batch-size 1 --epochs 1 --lora-rank 8 \
  --learning-rate 1e-4 --warmup-ratio 0.05 \
  --minimum-learning-rate-ratio 0.1 --checkpoint-steps 2000
```

Checkpoints are written under `<output_dir>/checkpoint` on the persistent disk and removed after the adapter and `run.json` are saved. Each checkpoint includes the full model and optimizer state, so it can take several gigabytes. If the run is interrupted, rerun the same command with the same output directory and settings; training resumes from the latest checkpoint and skips completed batches. Keep the checkpoint directory intact until the run finishes.

Reduce the sequence length or batch size if the server runs out of GPU memory. The command saves a LoRA adapter and `run.json` under the selected output directory when training finishes. Use persistent storage for the model cache, checkpoints, and final adapter.

The `data/` and `runs/` directories are not tracked by Git. For the first CPT pilot, transfer at least `data/processed/cpt-medical-v3/cpt_train.jsonl` and `cpt_validation.jsonl` into the matching paths under your project directory. Keep the raw archive, full processed splits, model cache, and run outputs on persistent storage or transfer them separately when moving between machines. Do not commit model weights or datasets to the code repository.

## Current limitations

- The earlier verified run was a two-training-example Qwen2.5 pipeline pilot on one RTX 2080 Ti; it does not establish Qwen3.5 compatibility, full-run time, or improved answer quality.
- The current v3 CPT data has been prepared, but the Qwen3.5 pilot has not run. The SFT training code and final evaluation code remain to be implemented.
- DPO and GRPO are optional future branches from the same SFT checkpoint; neither is implemented in this repository. DPO needs preference pairs, while GRPO needs a defined reward signal and a compatible training stack.
