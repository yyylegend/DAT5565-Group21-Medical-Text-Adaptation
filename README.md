# DAT5565 Group 21 — Medical Text Adaptation

**Languages:** English | [简体中文](README.zh-CN.md)

This README walks through the project from a new Linux GPU server to CPT, SFT, and foundational LSTM experiments. Run these commands in an SSH terminal connected to the server, not in Windows PowerShell. The project plan and data decisions are in the [research overview](docs/research/README.en.md).

## Which branch to use

| Name | Purpose |
|---|---|
| `main` | Stable completed CPT baseline; SFT/LSTM work has not been merged. |
| `runqi/sft-work` | Active course-project branch covering CPT, SFT, LSTM, export, and evaluation. Its existing name is retained for server checkouts. |
| `cpt-baseline-2026-09-28` | CPT baseline tag fixed at `5800729`, so it remains accessible after future updates to `main`. |

The installation instructions below clone the active work branch. For an existing checkout:

    git fetch origin
    git switch runqi/sft-work
    git pull --ff-only

CPT/SFT and LSTM have been compared on the same 200 test questions. Current work is the public MMLU subject evaluation, manual answer review, and report documentation. After runtime checks and result documentation, merge the course pipeline into `main` through one PR. Optional distillation, DPO, and GRPO experiments get separate branches.

TensorFlow training, export, and LSTM evaluation use `uv run healthcpt <command>`. Qwen checks and QA evaluation still use `python src/healthcpt/...` in the existing PyTorch/Transformers environment. The old `cpt-pilot` and LSTM module commands remain supported.

## 1. What you need

- A Linux x86_64 server with an NVIDIA GPU and a compatible driver.
- A persistent disk provided by your cloud service. Clone the repository there so the code, model cache, data, and checkpoints survive a container restart.
- Internet access for the first setup and model download.

The locked environment uses Python 3.12, TensorFlow 2.21, Keras 3.15, and KerasHub 0.32. The project requires Python 3.11 or 3.12.
For driver and GPU compatibility details, see the [TensorFlow installation guide](https://www.tensorflow.org/install/pip).

## 2. Install Git, tmux, and uv

Install the system tools only if they are missing. If your shell prompt shows that you are root, run apt-get without sudo:

    apt-get update
    apt-get install -y git tmux curl

If you are a regular user with sudo access, use:

    sudo apt-get update
    sudo apt-get install -y git tmux curl

Install uv with its official installer:

    curl -LsSf https://astral.sh/uv/install.sh | sh

Open a new SSH shell, then check:

    git --version
    tmux -V
    uv --version

If sudo is not installed and your prompt is root, that is normal: omit sudo. The [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) has other install options.

## 3. Clone the repository and install Python packages

Find the persistent-disk mount path in your cloud provider. It varies by server. Replace the first path below with that mount point:

    cd /path/to/your/persistent-disk
    git clone --branch runqi/sft-work https://github.com/yyylegend/DAT5565-Group21-Medical-Text-Adaptation.git
    cd DAT5565-Group21-Medical-Text-Adaptation
    uv sync --locked --python 3.12
    uv run python --version

The repository is public, so cloning it should not ask for a GitHub username or password. uv will use Python 3.12 for this project and can download it if the server does not already have it. See the [uv Python guide](https://docs.astral.sh/uv/guides/install-python/).

## 4. Put the data in the project

Git does not contain the datasets. If you already prepared the data on another computer, upload these files into the matching folder on the server:

    data/processed/cpt-medical-v3/cpt_train.jsonl
    data/processed/cpt-medical-v3/cpt_validation.jsonl

For later QA evaluation, also upload:

    data/processed/cpt-medical-v3/qa_validation_eval.jsonl
    data/processed/cpt-medical-v3/qa_test_eval.jsonl
    data/processed/cpt-medical-v3/manifest.json

SFT and LSTM training also require:

    data/processed/medquad-v1/qa_train.jsonl
    data/processed/medquad-v1/manifest.json

Create the destination folder before transferring files:

    mkdir -p data/processed/cpt-medical-v3 data/processed/medquad-v1

You can use your provider's file browser or scp. Keep the file names and folder structure shown above. The current CPT files contain 24,240 training text chunks and 1,645 validation chunks.

### Rebuild the data from the original sources

If you do not have prepared files, run the data pipeline on the server. These commands download the MedQuAD archive, prepare its QA splits, download MedlinePlus and a limited PMC sample, then create CPT and evaluation files.

    mkdir -p data/raw
    curl -L https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip \
      -o data/raw/medquad-master.zip

    uv run healthcpt audit-medquad data/raw/medquad-master.zip
    uv run healthcpt prepare-medquad \
      data/raw/medquad-master.zip \
      data/processed/medquad-v1

    uv run healthcpt download-medical-sources \
      data/processed/medquad-v1/qa_train.jsonl \
      data/raw/medical-sources-v3 \
      --topic-limit 20 --articles-per-topic 30

    uv run healthcpt prepare-medical-corpus \
      data/processed/medquad-v1 \
      data/raw/medical-sources-v3 \
      data/processed/cpt-medical-v3 \
      --medquad-cpt-limit 6000 --chunk-words 200

This downloads at most 600 English CC0/CC BY PMC articles selected from common MedQuAD topics; it does not download the whole PMC collection. The final manifest records source counts, licenses, hashes, and removed overlaps. Raw and processed datasets stay outside Git.
The upstream MedQuAD archive can change; compare the source hash in the generated manifest when matching counts matters.

## 5. Check the GPU and prepare the model cache

Run these commands from the project directory:

    nvidia-smi
    uv run python -c 'import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices("GPU"))'

The second command must show at least one GPU. If it prints an empty list, stop here and check the server image, driver, and GPU access.

The training command loads the model from Hugging Face on its first run. In the tmux session, set its cache to the ignored models folder before starting training. Keeping the repository on persistent storage keeps the cache there too. Hugging Face documents the [HF_HOME cache setting](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables).

## 6. Start a tmux session before training

Run training inside tmux. A normal SSH terminal can close or disconnect and stop a process running directly inside it. tmux keeps the training shell alive when the SSH client disconnects. See the [tmux getting-started guide](https://github.com/tmux/tmux/wiki/Getting-Started).

From the project directory, start a named session:

    tmux new -s healthcpt

You are now inside tmux. Set the model cache in this shell before the first model load, then launch the pilot or full run. The repository uses the Hugging Face Qwen3.5-2B-Base model in text-only mode.

    mkdir -p models/huggingface
    export HF_HOME="$PWD/models/huggingface"

### Optional small pilot

This short run checks model loading, GPU access, and adapter saving before the full dataset run:

    uv run healthcpt cpt-train \
      data/processed/cpt-medical-v3/cpt_train.jsonl \
      data/processed/cpt-medical-v3/cpt_validation.jsonl \
      runs/qwen3_5_2b_cpt_pilot \
      --preset hf://Qwen/Qwen3.5-2B-Base \
      --limit-train 32 --limit-validation 8 \
      --sequence-length 512 --batch-size 1 --epochs 1 --lora-rank 8

### Full CPT run

    uv run healthcpt cpt-train \
      data/processed/cpt-medical-v3/cpt_train.jsonl \
      data/processed/cpt-medical-v3/cpt_validation.jsonl \
      runs/qwen3_5_2b_cpt_full_1epoch \
      --preset hf://Qwen/Qwen3.5-2B-Base \
      --limit-train 24240 --limit-validation 1645 \
      --sequence-length 512 --batch-size 1 --epochs 1 --lora-rank 8 \
      --learning-rate 1e-4 --warmup-ratio 0.05 \
      --minimum-learning-rate-ratio 0.1 --checkpoint-steps 2000

At startup, the script prints the model, dataset sizes, training settings, output folder, TensorBoard folder, and checkpoint schedule. The Keras progress bar then shows the step, estimated time remaining (ETA), loss, and token accuracy. It refreshes in place on one terminal line. Validation loss appears after validation. These are language-model training metrics, not medical-answer accuracy.

The run writes TensorBoard logs every 100 steps under the run folder. In the attached tmux terminal, press Ctrl+B, release both keys, then press lowercase c to open another window. Run this command in the new window:

    uv run tensorboard \
      --logdir runs/qwen3_5_2b_cpt_full_1epoch/tensorboard \
      --host 127.0.0.1 --port 6006

Open port 6006 through an SSH tunnel or your cloud provider's port-forwarding feature.

## 7. Detach, reconnect, and keep the server running

To leave training running, press Ctrl+B, release both keys, then press lowercase d. This detaches from tmux; it does not stop the training process.

Later, reconnect to the server and run:

    cd /path/to/your/persistent-disk/DAT5565-Group21-Medical-Text-Adaptation
    tmux ls
    tmux attach -t healthcpt

Closing the local terminal after detaching is safe. Stopping, restarting, or releasing the cloud instance stops the GPU process; tmux does not keep compute running after the instance is shut down. The checkpoint remains on the persistent disk if it was saved.

Do not press Ctrl+C to detach. Ctrl+C interrupts the Python training process.

## 8. Resume after an interruption

The run saves a recovery checkpoint every 2,000 steps. To stop at a checkpoint, wait until the progress counter reaches a multiple of 2,000 and allow a few seconds for the disk write to finish before pressing Ctrl+C. If training is interrupted between checkpoints, start the same full command again with the same dataset, output folder, and settings. It resumes from the latest checkpoint; work since that checkpoint may need to run again.

Do not delete the run folder's checkpoint directory before training completes. After a successful run, the script saves the LoRA adapter and run.json, then removes the temporary checkpoint.

If you need to update the repository while training is in progress, wait until a checkpoint has been written, press Ctrl+C, pull the code, then rerun the same training command:

    git pull --ff-only origin main

The training settings and data hashes are checked when resuming. Keep the command and data files unchanged. A completed run has an adapter and run.json; export it instead of starting CPT again:

The exporter preserves the original multimodal components. First locate the full Base snapshot downloaded for training:

    find models/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots -mindepth 1 -maxdepth 1 -type d

The current run used snapshot b1485b2fa6dfa1287294f269f5fb618e03d52d7c. Use that local directory as --base-dir:

    uv run healthcpt export-hf runs/qwen3_5_2b_cpt_full_1epoch \
      --base-dir models/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c

For another run, replace the snapshot folder with the one printed by find. Use the cache path from the training environment if HF_HOME was elsewhere. The command reads that local Base and never downloads a newer version. The training record did not store the Base revision, so keep the selected snapshot with the export record.

The default output is hf_export_multimodal inside the run folder; it must be empty. Check free space with df -h . first; the output needs roughly one more full model copy. The exporter copies the original Safetensors files and replaces only the 12 text q/v projection tensors trained by LoRA. It keeps the original tensor dtypes, shard index, full config, image/video processor, tokenizer, and license files. It checks by SHA-256 that every other weight byte, including the vision encoder and visual merger, is unchanged. export_report.json records the selected Base snapshot and these checks. Keep the original adapter too.

Preserving vision weights does not establish unchanged image/video quality: text CPT changes the language model that interprets visual features. For this run, the Hugging Face export loaded all 617 weights in Transformers, and both text and image inference completed with a generated solid-color test image. This confirms that the export loads and accepts image input; it does not measure medical-answer quality or real-image understanding.

For future exports, use a local JPG/PNG for this separate Transformers loading and inference check:

    uv run --no-project --python 3.12 --with "transformers>=5.9,<6" --with torch --with torchvision --with pillow \
      python src/healthcpt/verify_hf.py \
      runs/qwen3_5_2b_cpt_full_1epoch/hf_export_multimodal \
      --image /path/to/test-image.jpg

This installs optional inference packages outside the training environment. PyTorch and its CUDA packages can be large downloads. If your active Python environment already has a Qwen3.5-compatible Transformers version, PyTorch, and Pillow, run the script with `python` directly instead. The check looks for missing or unexpected weights, finite logits, and short text/image generation, then writes inference_check.json. It is a compatibility check, not a quality benchmark. For quality evaluation, compare the original Base and merged model on the same text and image examples.

## 9. Common messages and errors

| Console message | Meaning and action |
|---|---|
| git: command not found or tmux: command not found | Install the missing tool with apt-get. If sudo is missing and the prompt is root, run apt-get without sudo. |
| uv: command not found | Install uv, then open a new shell and check uv --version. |
| GitHub asks for a username/password during clone | The repository is public. Cancel with Ctrl+C and use the plain HTTPS clone URL above. |
| All log messages before absl::InitializeLog() or cpu_feature_guard information | TensorFlow startup/logging information. It is safe to ignore if training continues. |
| hwloc invalid topology or Failed to find hwloc NUMA node | Container CPU/NUMA metadata warning. It is safe to ignore if TensorFlow lists a GPU and training steps advance. |
| Hugging Face Hub unauthenticated-request warning | Public model downloads can continue without a token. A token may help if the Hub rate-limits the download. |
| KerasHub reports 297 weights not ported, all names begin with model.visual | Expected for this text-only run; those are the unused vision weights. If any unported name is outside model.visual, stop and inspect the model load. |
| TensorFlow lists no GPU or says No TensorFlow GPU detected | Do not ignore. Check nvidia-smi, the selected server image, and whether the container can access the GPU. |
| CUDA out of memory or process is Killed | Do not ignore. Check available GPU memory and reduce sequence length or batch size before rerunning. |
| KeyboardInterrupt after pressing Ctrl+C | The current run was interrupted intentionally. Rerun the same command to restore from the latest checkpoint. |

The current script prints the checkpoint frequency and location at startup, then saves quietly so checkpoint messages do not interrupt the live progress bar. The progress bar itself refreshes on one line by design.

## 10. Run a text QA comparison

The optional evaluator compares the Base model with one trained model on the same MedQuAD questions. Run it from a Python environment that already has PyTorch and a Qwen3.5-compatible Transformers version. Start with 50 validation questions to check the setup:

    python src/healthcpt/evaluate_qa.py \
      --base-dir models/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
      --candidate-dir runs/qwen3_5_2b_cpt_full_1epoch/hf_export_multimodal \
      --candidate-name CPT \
      --qa-file data/processed/cpt-medical-v3/qa_validation_eval.jsonl \
      --output-dir runs/qa_eval_cpt_validation_pilot \
      --limit 50

Both models receive the same `Question: ...\nAnswer:` prompt and greedy decoding settings. Remove `--limit 50` to compare all validation questions. The script writes per-question answers to predictions.jsonl and summary metrics to metrics.json. It reports normalized exact match, token F1, and ROUGE-L. These compare text overlap with reference answers; they do not establish medical correctness. Review answers manually. Validation is for development comparisons; a seeded 200-question test sample has now been used to compare Base, CPT, and CPT+SFT, while the full 1,573-row test file remains unevaluated. See the [research overview](docs/research/README.en.md) for those results and limits. Choose a new, empty output folder for each run.

## 11. Run SFT

SFT continues from a completed CPT run. It reloads the same Base model and CPT LoRA adapter, then trains on MedQuAD question-answer pairs. Start with a small pilot to check the software setup and GPU memory:

    uv run healthcpt sft-train \
      runs/qwen3_5_2b_cpt_full_1epoch \
      data/processed/medquad-v1/qa_train.jsonl \
      data/processed/cpt-medical-v3/qa_validation_eval.jsonl \
      runs/qwen3_5_2b_sft_pilot \
      --limit-train 128 --limit-validation 32 \
      --sequence-length 512 --batch-size 1 --learning-rate 2e-5

If the pilot runs correctly, train for one epoch on all 12,799 training pairs and 1,471 validation pairs:

    uv run healthcpt sft-train \
      runs/qwen3_5_2b_cpt_full_1epoch \
      data/processed/medquad-v1/qa_train.jsonl \
      data/processed/cpt-medical-v3/qa_validation_eval.jsonl \
      runs/qwen3_5_2b_sft_full_1epoch \
      --limit-train 12799 --limit-validation 1471 \
      --sequence-length 512 --batch-size 1 \
      --learning-rate 2e-5 --warmup-ratio 0.05 \
      --minimum-learning-rate-ratio 0.1 --checkpoint-steps 1000

The progress bar shows steps, ETA, loss, and token accuracy. TensorBoard logs go to `tensorboard/` inside the run folder. After an interruption, rerun the exact same command with the same folder to resume from the latest checkpoint. Each input is the full `Question: ...\nAnswer: ...` text, and KerasHub calculates next-token loss on all non-padding question and answer tokens. The sequence length is fixed at 512, so long examples are truncated and should be considered when interpreting results.

After training, `sft_adapter.lora.h5` contains the continued CPT+SFT LoRA updates; `run.json` and `training_config.json` record the run. The adapter is not a standalone model. Export the full Hugging Face multimodal model using the same Base snapshot as CPT:

    uv run healthcpt export-hf runs/qwen3_5_2b_sft_full_1epoch \
      --base-dir models/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
      --output-dir runs/qwen3_5_2b_sft_full_1epoch/hf_export_multimodal

The export folder contains the Hugging Face configuration, tokenizer/processor files, and Safetensors weights; vision-related weights are copied from the original Base model. The training adapter is `.h5`; the complete export uses Safetensors. Then run `evaluate_qa.py` with `--candidate-dir` pointing to this `hf_export_multimodal` folder and `qa_test_eval.jsonl` for the final Base-versus-CPT+SFT comparison.

## 12. Required foundational LSTM baseline

The instructor approved at least 10,000 medical text records or QA pairs and requires primary TensorFlow/Keras experiments with a foundational architecture baseline. `lstm_baseline.py` trains `Embedding → forward LSTM → Dense` from scratch on the same QA training split as SFT. Its vocabulary is learned only from training data: up to 20,000 words, embedding dimension 128, and hidden size 256, totaling about 8.1 million parameters at the vocabulary limit. It can generate answers; it is not expected to match the knowledge or fluency of a pretrained SLM.

Start with a pilot inside tmux:

    uv run healthcpt lstm-train \
      --train-file data/processed/medquad-v1/qa_train.jsonl \
      --validation-file data/processed/cpt-medical-v3/qa_validation_eval.jsonl \
      --output-dir runs/lstm_qa_pilot \
      --limit-train 128 --limit-validation 32 --epochs 1

Then train on all pairs for at most five epochs with validation-loss early stopping:

    uv run healthcpt lstm-train \
      --train-file data/processed/medquad-v1/qa_train.jsonl \
      --validation-file data/processed/cpt-medical-v3/qa_validation_eval.jsonl \
      --output-dir runs/lstm_qa_full \
      --vocab-size 20000 --embedding-dim 128 --hidden-size 256 \
      --sequence-length 512 --batch-size 8 --epochs 5 --learning-rate 1e-3

The best full model (`model.keras`) and latest model (`latest.keras`) are saved after epochs, alongside the vocabulary, training settings, loss curves, and unknown-word/truncation statistics. There is no step-resume command for this baseline; keep the tmux session and server running. These are Keras models, not Hugging Face Safetensors.

Evaluate the same 200 test questions and reuse the saved Qwen answers:

    uv run healthcpt lstm-evaluate \
      --model-dir runs/lstm_qa_full \
      --qa-file data/processed/cpt-medical-v3/qa_test_eval.jsonl \
      --output-dir runs/qa_eval_lstm_test_200 --limit 200 --seed 5565 \
      --cached-qwen-predictions runs/qa_eval_cpt_sft_test_200/predictions.jsonl

The script verifies question, reference, source-line, and data-hash alignment before computing the same text-overlap metrics for LSTM, Base, and CPT+SFT. LSTM word tokens and Qwen subword tokens differ, so their 512-token input and 128-token generation caps are not equivalent; disclose this in the report. LSTM word perplexity cannot be directly compared with Qwen perplexity. This comparison measures differences between complete model setups, including model size and pretraining exposure, rather than isolating architecture alone.

## 13. Automatic accuracy on a public benchmark

The additional benchmark is the complete `professional_medicine` subject from [MMLU](https://github.com/hendrycks/test): 272 four-choice test questions with five dev questions as prompt demonstrations. Report **MMLU professional_medicine, 5-shot accuracy**, not the full MMLU aggregate score. It measures medical exam knowledge and supplements the existing MedQuAD held-out text-overlap evaluation.

Prepare only this subject in the existing PyTorch/Transformers inference environment, with no additional dataset library:

    python src/healthcpt/evaluate_mmlu.py prepare

The source is `cais/mmlu`. The script verifies a fixed dataset revision and row counts and saves dev/test JSONL plus a hash manifest. It does not download the full MMLU collection. Alternatively, transfer the prepared `data/processed/mmlu-professional-medicine/` directory to the server.

For an execution pilot, add `--limit 10` and choose another output folder. Run the full evaluation inside tmux:

    python src/healthcpt/evaluate_mmlu.py evaluate \
      --base-dir models/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
      --candidate-dir runs/qwen3_5_2b_sft_full_1epoch/hf_export_multimodal \
      --output-dir runs/mmlu_professional_medicine_5shot

The script selects the largest next-token probability among ` A`, ` B`, ` C`, and ` D`; it does not generate long responses or require a paid judge. Both models see the same 272 questions and five dev demonstrations. Test gold labels never enter prompts. `metrics.json` records accuracy, correct counts, paired outcomes, and settings; per-question predictions are saved as JSONL. Download, prompt, and tokenizer checks passed; server GPU scoring remains pending. Public benchmark content may have appeared in pretrained models' training data, so the score does not establish clinical safety or a completely contamination-free evaluation.

## Project files

- data/: raw sources and processed datasets; not tracked by Git.
- models/: downloaded model cache; not tracked by Git.
- runs/: checkpoints, TensorBoard logs, adapters, and run summaries; not tracked by Git.
- docs/research/: bilingual project overview and research notes.

Read the Python files by responsibility; all live under `src/healthcpt/`:

| Step | Files |
|---|---|
| Command entry | `cli.py` |
| Data download and preparation | `medquad.py`, `medical_data.py` |
| Training | `cpt.py`, `sft.py`, `lstm_baseline.py` |
| Weight merging and file checks | `export_hf.py`, `checkpoint_files.py` |
| Qwen checks and evaluation | `verify_hf.py`, `evaluate_qa.py`, `evaluate_mmlu.py` |
| Shared QA sampling and metrics | `qa_metrics.py` |

Qwen3.5-2B CPT and SFT runs are complete. The CPT Hugging Face export passed a basic Transformers text/image check; the SFT export loaded for QA text evaluation. Seeded 200-question validation/test evaluations and the LSTM comparison are recorded; the full test file, public MMLU subject evaluation, and manual answer review remain pending. See the [research overview](docs/research/README.en.md) for run settings, results, and limits. DPO and GRPO remain optional and are not implemented.
