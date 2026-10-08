# DAT5565 第 21 组：医学文本适配

**语言 / Language:** 简体中文 | [English](README.md)

这份 README 从新服务器开始，带你配置仓库、准备数据，并在 GPU 服务器上用 tmux 运行 CPT、SFT 和 LSTM 基础模型。下面的命令要在 SSH 连接到服务器后的 Linux 终端运行，不是在 Windows PowerShell 中运行。项目研究计划和数据来源见[研究概览](docs/research/README.md)。

## 当前应该用哪个分支

| 名称 | 用途 |
|---|---|
| `main` | 课程项目主线，包含 CPT、SFT、LSTM、导出、评测和已执行的 Notebook。 |
| `runqi/sft-work` | 保留给服务器已有 checkout 的开发分支；后续新工作从 `main` 开始。 |
| `cpt-baseline-2026-09-28` | 固定在 `5800729` 的 CPT 基线标签，方便未来更新 `main` 后仍能回到这个版本。 |

下面的安装指引克隆 `main`。已有 checkout 的工作区干净时，可执行以下命令切换到课程主线：

    git fetch origin
    git switch main
    git pull --ff-only

最终自动评测已记录：Base、CPT、CPT+SFT 和 LSTM 使用同一批固定种子的 200 道 QA 测试题；完整的 MMLU professional_medicine 272 题也已完成。Base/CPT+SFT 的 EOS 修正版在 `runs/qa_eval_cpt_sft_test_200_eosfix_wsl/`，使用 RTX 2080 Ti；早期 RTX 4090 Infra 运行保留作历史记录。EOS 修正后，部分回答仍重复并达到 128-token 上限。1,573 道 QA 测试题尚未全量评测，人工抽查仍待完成。报告已按提交稿修订并在本地编译为 PDF；课程要求 Word 报告，可解释性、交叉验证、部署说明、10 分钟展示视频及团队调查等交付项也要跟进。详见[提交清单](docs/SUBMISSION_CHECKLIST.md)。DPO、GRPO 尚未实现。

TensorFlow 的训练、导出和 LSTM 评测统一用 `uv run healthcpt <命令>`；Qwen 图文检查和问答评测仍在已有 PyTorch/Transformers 环境运行 `python src/healthcpt/...`。旧的 `cpt-pilot` 和 LSTM 模块命令继续可用。

## 1. 开始前需要准备

- Linux x86_64 GPU 服务器，能使用 NVIDIA GPU 和兼容的驱动。
- 云平台提供的持久磁盘。仓库应克隆到持久磁盘，而不是容器临时盘。
- 首次安装依赖和下载模型时需要网络。

项目锁定的环境使用 Python 3.12、TensorFlow 2.21、Keras 3.15 和 KerasHub 0.32；项目支持 Python 3.11 或 3.12。
驱动和 GPU 兼容性可参考 [TensorFlow 安装指南](https://www.tensorflow.org/install/pip)。

## 2. 安装 Git、tmux 和 uv

如果命令不存在，再安装系统工具。若命令行提示符是 root（通常以 # 结尾），直接运行 apt-get，不需要 sudo：

    apt-get update
    apt-get install -y git tmux curl

若你使用普通用户并且有 sudo 权限，则运行：

    sudo apt-get update
    sudo apt-get install -y git tmux curl

用官方安装脚本安装 uv：

    curl -LsSf https://astral.sh/uv/install.sh | sh

安装后重新打开 SSH 终端，再检查：

    git --version
    tmux -V
    uv --version

如果你看到 sudo: command not found，而提示符是 root，这是正常的；去掉命令里的 sudo 即可。更多方式见 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)。

## 3. 克隆仓库并安装 Python 依赖

先在云平台确认持久磁盘的挂载路径。不同服务器的路径不同。把下面第一行替换成你的实际路径：

    cd /path/to/your/persistent-disk
    git clone --branch main https://github.com/yyylegend/DAT5565-Group21-Medical-Text-Adaptation.git
    cd DAT5565-Group21-Medical-Text-Adaptation
    uv sync --locked --python 3.12
    uv run python --version

仓库是公开的，克隆时不需要 GitHub 用户名或密码。uv 会使用 Python 3.12；如果服务器没有安装，它可以自动下载。详见 [uv 的 Python 安装说明](https://docs.astral.sh/uv/guides/install-python/)。

## 4. 把数据放进仓库

数据集不在 Git 里。如果你已经在其他电脑上清理好了数据，把下面文件上传到服务器对应目录：

    data/processed/cpt-medical-v3/cpt_train.jsonl
    data/processed/cpt-medical-v3/cpt_validation.jsonl

后续运行问答评测还需要：

    data/processed/cpt-medical-v3/qa_validation_eval.jsonl
    data/processed/cpt-medical-v3/qa_test_eval.jsonl
    data/processed/cpt-medical-v3/manifest.json

SFT 和 LSTM 训练还需要：

    data/processed/medquad-v1/qa_train.jsonl
    data/processed/medquad-v1/manifest.json

上传前可以先创建目标目录：

    mkdir -p data/processed/cpt-medical-v3 data/processed/medquad-v1

你可以使用云平台的文件管理器或 scp。保持上面的文件名和目录结构。当前 CPT 数据包含 24,240 条训练文本片段和 1,645 条验证片段。

### 从原始数据重新生成

如果没有已经处理好的文件，可以在服务器运行下面的数据流程：下载 MedQuAD 压缩包，准备问答划分，下载 MedlinePlus 和有限数量的 PMC 文章，然后生成 CPT 和评测文件。

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

PMC 命令最多抽取 600 篇英文 CC0/CC BY 开放获取文章，不会下载整个 PMC 数据库。最终清单会记录数据来源、许可、哈希和移除的重叠内容。原始数据和处理后的数据都不会提交到 Git。
MedQuAD 上游压缩包可能变化；如果需要复现统计数字，请对照生成清单里的来源哈希。

## 5. 检查 GPU，并设置模型缓存

在仓库目录中运行：

    nvidia-smi
    uv run python -c 'import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices("GPU"))'

第二条命令必须列出至少一个 GPU。如果结果为空，先检查服务器镜像、驱动和容器的 GPU 访问权限。

训练命令会在第一次运行时从 Hugging Face 加载模型。启动 tmux 后，在 tmux 会话里把缓存设到被 Git 忽略的 models 目录。只要仓库在持久磁盘上，模型缓存也会保留。Hugging Face 对 [HF_HOME 缓存位置](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)有说明。

## 6. 必须在 tmux 里启动训练

请在 tmux 会话中运行训练。若直接在普通 SSH 终端里启动，远程连接中断或关闭终端时，训练进程可能一起停止。tmux 可以在 SSH 客户端断开后保留服务器上的训练会话，详见 [tmux 入门说明](https://github.com/tmux/tmux/wiki/Getting-Started)。

在仓库目录启动一个名为 healthcpt 的会话：

    tmux new -s healthcpt

现在你已经进入 tmux。先设置模型缓存，再运行试跑或全量训练。仓库使用 Hugging Face 上的 Qwen3.5-2B-Base，并且只训练文本部分。

    mkdir -p models/huggingface
    export HF_HOME="$PWD/models/huggingface"

### 可选：先做小规模试跑

这会检查模型能否加载、GPU 是否可用，以及 adapter 能否保存：

    uv run healthcpt cpt-train \
      data/processed/cpt-medical-v3/cpt_train.jsonl \
      data/processed/cpt-medical-v3/cpt_validation.jsonl \
      runs/qwen3_5_2b_cpt_pilot \
      --preset hf://Qwen/Qwen3.5-2B-Base \
      --limit-train 32 --limit-validation 8 \
      --sequence-length 512 --batch-size 1 --epochs 1 --lora-rank 8

### 全量 CPT 训练

    uv run healthcpt cpt-train \
      data/processed/cpt-medical-v3/cpt_train.jsonl \
      data/processed/cpt-medical-v3/cpt_validation.jsonl \
      runs/qwen3_5_2b_cpt_full_1epoch \
      --preset hf://Qwen/Qwen3.5-2B-Base \
      --limit-train 24240 --limit-validation 1645 \
      --sequence-length 512 --batch-size 1 --epochs 1 --lora-rank 8 \
      --learning-rate 1e-4 --warmup-ratio 0.05 \
      --minimum-learning-rate-ratio 0.1 --checkpoint-steps 2000

开始时，脚本会分行显示模型、数据量、训练参数、输出目录、TensorBoard 目录和检查点频率。随后 Keras 进度条显示 step、预计剩余时间（ETA）、loss 和 token accuracy。进度条在同一行动态刷新，这是正常的；验证结束后会显示验证 loss。这些是语言模型训练指标，不代表医疗问答正确率。

训练日志每 100 步写入运行目录下的 TensorBoard 文件。查看曲线时，在 tmux 窗口按 Ctrl+B，松开按键，再按小写 c 新建窗口，然后在新窗口运行：

    uv run tensorboard \
      --logdir runs/qwen3_5_2b_cpt_full_1epoch/tensorboard \
      --host 127.0.0.1 --port 6006

通过 SSH 隧道或云平台端口转发功能访问 6006 端口。

## 7. 暂离终端，再回来查看训练

离开训练时，按 Ctrl+B，松开按键，再按小写 d。这会从 tmux 分离，但训练仍在服务器上运行。

之后重新 SSH 登录并运行：

    cd /path/to/your/persistent-disk/DAT5565-Group21-Medical-Text-Adaptation
    tmux ls
    tmux attach -t healthcpt

分离 tmux 后关闭本地终端是安全的。云服务器本身仍需保持运行；关闭、重启或释放实例会停止 GPU 进程。已经保存的检查点仍在持久磁盘上。

不要用 Ctrl+C 来暂离；Ctrl+C 会中断 Python 训练程序。

## 8. 中断后继续训练

训练每 2,000 步保存一次恢复检查点。若要在检查点处停止，等进度计数到 2,000 的倍数后，再留几秒让磁盘写入完成，然后按 Ctrl+C。如果在两次检查点之间中断，用相同数据、输出目录和参数重新运行全量训练命令，脚本会从最近的检查点继续；上次检查点之后的步骤可能需要重跑。

训练成功结束前，不要删除运行目录里的 checkpoint 文件夹。训练完成后，脚本会保存 LoRA adapter 和 run.json，再清理临时检查点。

如果训练时要更新代码，等到检查点保存完成后按 Ctrl+C 停止训练，再拉取代码并用原命令继续：

    git pull --ff-only origin main

恢复时，数据文件和训练参数都必须与之前一致。训练已经成功完成后，不要再次运行 CPT 命令；使用导出命令即可：

导出脚本会保留基座原有的多模态组件。先找到训练时下载的完整 Base 快照：

    find models/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots -mindepth 1 -maxdepth 1 -type d

这次训练使用的快照是 b1485b2fa6dfa1287294f269f5fb618e03d52d7c。用它作为 --base-dir：

    uv run healthcpt export-hf runs/qwen3_5_2b_cpt_full_1epoch \
      --base-dir models/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c

其他训练请把快照目录换成 find 命令列出的正确版本。如果训练时 HF_HOME 指向其他位置，请使用对应缓存路径。训练记录没有保存模型 revision，所以请保留本次使用的 Base 快照信息。导出命令只读取指定的本地 Base，不会下载新版本覆盖它。

默认输出到运行目录下的 hf_export_multimodal，目标目录必须为空。先用 df -h . 检查磁盘空间；导出约需再存一份完整模型。脚本复制原始 Safetensors 文件，只替换 LoRA 训练过的 12 个文本 q/v 投影权重。权重精度、分片索引、完整配置、图像和视频 processor、tokenizer 及许可文件都会保留。脚本会用 SHA-256 检查其余权重字节（包括视觉编码器和视觉连接层）完全未变，并把 Base 快照信息和检查结果写入 export_report.json。原 adapter 也请保留。

保留视觉权重并不代表图像或视频效果没有变化：CPT 调整了负责理解视觉特征的语言模型。本次导出的 Hugging Face 模型已在 Transformers 中成功加载全部 617 项权重，并用生成的纯色测试图跑通了文本和图片推理。这只能确认模型能加载并接收图片输入，不能说明医学回答质量或真实图片理解能力保持不变。

以后导出模型后，可以选一张本地 JPG/PNG，在独立环境中用 Transformers 检查：

    uv run --no-project --python 3.12 --with "transformers>=5.9,<6" --with torch --with torchvision --with pillow \
      python src/healthcpt/verify_hf.py \
      runs/qwen3_5_2b_cpt_full_1epoch/hf_export_multimodal \
      --image /path/to/test-image.jpg

这会在训练环境之外安装可选的推理依赖。PyTorch 及其 CUDA 组件可能需要下载较大的文件；如果当前 Python 环境已经有支持 Qwen3.5 的 Transformers、PyTorch 和 Pillow，可以直接用 `python` 运行脚本。检查会确认权重没有缺失或多余、输出数值正常，并分别生成一小段文本和图片描述，结果写入 inference_check.json。这只能确认加载和推理能跑通；要比较效果，需要让原始 Base 和合并模型回答同一批文本、图片问题。

## 9. 常见提示和错误

| 控制台信息 | 含义和处理方法 |
|---|---|
| git: command not found 或 tmux: command not found | 用 apt-get 安装缺少的工具。若当前是 root 用户，不要加 sudo。 |
| uv: command not found | 按上面的步骤安装 uv，然后重新打开 SSH 终端。 |
| 克隆时 GitHub 要求用户名或密码 | 仓库是公开的。按 Ctrl+C 取消，然后使用上面的纯 HTTPS 地址，不要输入密码。 |
| All log messages before absl::InitializeLog() 或 cpu_feature_guard 提示 | TensorFlow 启动和日志提示。只要训练继续运行，可以忽略。 |
| hwloc topology 警告或 Failed to find hwloc NUMA node | 容器提供的 CPU/NUMA 信息不完整。如果 TensorFlow 能看到 GPU 且 step 在增加，通常可以忽略。 |
| Hugging Face Hub unauthenticated-request 提示 | 公开模型仍可下载；这个提示主要是说未登录时请求速度可能受限。 |
| KerasHub 提示 297 个权重未加载，且名字都以 model.visual 开头 | 当前训练不包含视觉编码器，这是预期行为。若未加载权重里有其他名字，不要忽略，应先检查模型加载。 |
| TensorFlow 没列出 GPU 或提示 No TensorFlow GPU detected | 不能忽略。检查 nvidia-smi、服务器镜像和容器 GPU 权限。 |
| CUDA out of memory 或进程显示 Killed | 不能忽略。检查显存，降低序列长度或 batch size 后再运行。 |
| 按 Ctrl+C 后出现 KeyboardInterrupt | 这是主动中断时的正常提示。用相同命令重跑即可从最近的检查点恢复。 |

新版脚本会在开头显示检查点频率和位置，保存时不额外打印提示行，以免打断实时进度条。step、ETA 和 loss 在同一行动态变化是正常的。

## 10. 做一次文本问答对比

评测脚本会让 Base 模型和一个训练后的模型回答同一批 MedQuAD 问题。请在已经安装 PyTorch 和支持 Qwen3.5 的 Transformers 环境中运行。先抽 50 道验证题检查流程：

    python src/healthcpt/evaluate_qa.py \
      --base-dir models/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
      --candidate-dir runs/qwen3_5_2b_cpt_full_1epoch/hf_export_multimodal \
      --candidate-name CPT \
      --qa-file data/processed/cpt-medical-v3/qa_validation_eval.jsonl \
      --output-dir runs/qa_eval_cpt_validation_pilot \
      --limit 50

两个模型使用相同的 `Question: ...\nAnswer:` 提示词和贪心生成设置。去掉 `--limit 50` 可评测全部验证题。脚本会把每题的模型回答写入 `predictions.jsonl`，把汇总结果写入 `metrics.json`，包括 normalized exact match、token F1 和 ROUGE-L。这些指标衡量回答与参考答案的文字重合度，不能证明医学正确性；还要人工抽查。验证集用于开发阶段比较；当前已在测试集抽取 200 题做 Base、CPT 和 CPT+SFT 对比，全量 1,573 题尚未运行。测试集抽样结果及限制见[研究概览](docs/research/README.md)。每次运行请使用一个新的空输出目录。

如果要对同一批 200 道测试题完成 Base 与 CPT+SFT 的最终对比，脚本还会记录模型加载时间、单题延迟、生成速度、显存峰值和模型目录文件大小。添加 `--bertscore` 可额外计算 BERTScore F1。先在当前 PyTorch 评测环境安装 `bert-score`；它会下载约 1.4 GB 的 `roberta-large` 评分模型。BERTScore 在 Qwen 推理完成后才加载，因此它占用的显存不会计入 Qwen 的显存峰值。

    python -m pip install bert-score

    python src/healthcpt/evaluate_qa.py \
      --base-dir models/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
      --candidate-dir runs/qwen3_5_2b_sft_full_1epoch/hf_export_multimodal \
      --candidate-name CPT+SFT \
      --qa-file data/processed/cpt-medical-v3/qa_test_eval.jsonl \
      --output-dir runs/qa_eval_cpt_sft_test_200_eosfix_wsl \
      --limit 200 --seed 5565 --max-new-tokens 128 \
      --warmup-questions 3 --bertscore

评测同时把 tokenizer EOS 和 Qwen3.5 assistant-turn 标记 `<|im_end|>` 设为停止 token。EOS 修正版的 WSL 结果保存在 `runs/qa_eval_cpt_sft_test_200_eosfix_wsl/`。单题延迟不含模型加载时间，包含提示词处理、生成和解码；生成速度只按生成阶段计算。显存来自 PyTorch CUDA 分配器的统计；可以用 `nvidia-smi` 另行核对整张卡的占用。比较时保持 GPU、模型文件、题目和生成设置一致。正式跑 200 题前，先把命令中的 `--limit 200` 改为 `--limit 5`，并换一个输出目录，确认 BERTScore 能在当前 Transformers 环境正常加载。

## 11. 运行 SFT

SFT 从已完成的 CPT 运行目录继续训练；脚本会重新加载同一个 Base 和 CPT LoRA adapter，再用 MedQuAD 问答训练。先跑小样本，确认软件、显存和输出正常：

    uv run healthcpt sft-train \
      runs/qwen3_5_2b_cpt_full_1epoch \
      data/processed/medquad-v1/qa_train.jsonl \
      data/processed/cpt-medical-v3/qa_validation_eval.jsonl \
      runs/qwen3_5_2b_sft_pilot \
      --limit-train 128 --limit-validation 32 \
      --sequence-length 512 --batch-size 1 --learning-rate 2e-5

小样本正常后，用全部 12,799 条训练问答和 1,471 条验证问答运行一个 epoch：

    uv run healthcpt sft-train \
      runs/qwen3_5_2b_cpt_full_1epoch \
      data/processed/medquad-v1/qa_train.jsonl \
      data/processed/cpt-medical-v3/qa_validation_eval.jsonl \
      runs/qwen3_5_2b_sft_full_1epoch \
      --limit-train 12799 --limit-validation 1471 \
      --sequence-length 512 --batch-size 1 \
      --learning-rate 2e-5 --warmup-ratio 0.05 \
      --minimum-learning-rate-ratio 0.1 --checkpoint-steps 1000

提示符里的进度条会显示 step、ETA、loss 和 token accuracy。TensorBoard 日志保存在运行目录下的 `tensorboard/`；训练中断后，使用完全相同的命令和目录即可从最近一次检查点继续。SFT 输入是一条完整的 `Question: ...\nAnswer: ...` 文本，KerasHub 会对问题和答案的非填充 token 计算下一个 token 损失。序列长度固定为 512；过长问答会被截断，因此正式结果要结合长答案截断风险解读。

训练成功后，运行目录里的 `sft_adapter.lora.h5` 保存继续训练后的 CPT+SFT LoRA 更新，`run.json` 和 `training_config.json` 记录运行信息。它还不是独立模型。要导出完整的 Hugging Face 多模态模型，请使用训练时相同的 Base 快照：

    uv run healthcpt export-hf runs/qwen3_5_2b_sft_full_1epoch \
      --base-dir models/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
      --output-dir runs/qwen3_5_2b_sft_full_1epoch/hf_export_multimodal

导出目录使用 Hugging Face Transformers 可读取的配置、tokenizer/processor 文件和 Safetensors 权重；视觉相关权重沿用原始 Base。训练 adapter 是 `.h5`，完整导出才是 Safetensors。导出后可用上一节的 `evaluate_qa.py`，将 `--candidate-dir` 指向新的 `hf_export_multimodal`，并用 `qa_test_eval.jsonl` 做最终 Base 与 CPT+SFT 对比。

## 12. 教授要求的 LSTM 基础模型

教授已批准至少 10,000 条医学文本或问答对的数据量，并要求主要实验使用 TensorFlow/Keras，补充课堂基础架构对照。`lstm_baseline.py` 从零训练 `Embedding → 单向 LSTM → Dense`，使用与 SFT 相同的训练问答。词表只由训练集建立，默认最多 20,000 个词、embedding 128、LSTM hidden size 256；模型最大约 810 万参数。它能生成回答，但不预期具备预训练 SLM 的知识和语言能力。

先在 tmux 中运行小样本检查：

    uv run healthcpt lstm-train \
      --train-file data/processed/medquad-v1/qa_train.jsonl \
      --validation-file data/processed/cpt-medical-v3/qa_validation_eval.jsonl \
      --output-dir runs/lstm_qa_pilot \
      --limit-train 128 --limit-validation 32 --epochs 1

正式训练使用全部问答，最多 50 个 epoch，并按验证 loss 提前停止（patience 为 2）：

    uv run healthcpt lstm-train \
      --train-file data/processed/medquad-v1/qa_train.jsonl \
      --validation-file data/processed/cpt-medical-v3/qa_validation_eval.jsonl \
      --output-dir runs/lstm_qa_50epochs \
      --vocab-size 20000 --embedding-dim 128 --hidden-size 256 \
      --sequence-length 512 --batch-size 8 --epochs 50 --learning-rate 1e-3

本次正式运行在第 16 轮早停，验证 loss 最低的第 14 轮被保存为 `model.keras`；`latest.keras` 保存最后一轮。词表、训练参数、loss 曲线和未知词/截断统计也保存在运行目录。这里没有逐步断点恢复入口；tmux 会话和服务器应在训练过程中保持运行。模型格式是 Keras `.keras`，不是 Hugging Face Safetensors。

训练完成后，在相同 200 道测试题上评测，并复用之前保存的 Qwen 回答，避免重新跑大模型：

    uv run healthcpt lstm-evaluate \
      --model-dir runs/lstm_qa_50epochs \
      --qa-file data/processed/cpt-medical-v3/qa_test_eval.jsonl \
      --output-dir runs/qa_eval_lstm_16epoch_test_200 --limit 200 --seed 5565 \
      --cached-qwen-predictions runs/qa_eval_cpt_sft_test_200/predictions.jsonl

脚本核对缓存中的题目、参考答案、来源行号和数据哈希，再输出 LSTM、Base 和 CPT+SFT 的相同文字重合指标。LSTM 使用 128 个生成词，Qwen 使用最多 128 个子词；两者长度预算不同。模型规模和预训练数据也不同，因此结果反映的是完整模型方案差异，不能单独归因于 LSTM/Transformer 架构。

## 13. 公开 benchmark 的自动准确率

补充评测选用 [MMLU](https://github.com/hendrycks/test) 的 `professional_medicine` 完整子集：272 道四选一 test 题，使用 5 道 dev 题作为提示中的示例。分数是正确题数除以 272，名称应写作 **MMLU professional_medicine, 5-shot accuracy**，不是整个 MMLU 的总分。它测医学考试知识；此前 MedQuAD 留出题上的文字重合评测仍保留。

在已有 PyTorch/Transformers 推理环境中，先准备这一科目的小数据文件，不需要安装新的数据工具：

    python src/healthcpt/evaluate_mmlu.py prepare

下载来源是 `cais/mmlu`，脚本核对固定数据 revision 和行数，保存 test/dev JSONL 与哈希清单。只读取这一科目，不下载整个 MMLU。若其他电脑已准备好这三个文件，也可将 `data/processed/mmlu-professional-medicine/` 传到服务器。

先加 `--limit 10` 并使用不同输出目录做运行检查。完整评测命令如下，建议在 tmux 内运行：

    python src/healthcpt/evaluate_mmlu.py evaluate \
      --base-dir models/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
      --candidate-dir runs/qwen3_5_2b_sft_full_1epoch/hf_export_multimodal \
      --output-dir runs/mmlu_professional_medicine_5shot

MMLU 完整科目的 GPU 评测已完成：脚本比较 ` A`、` B`、` C`、` D` 四个答案 token 的下一 token 概率，不生成长答案。两模型使用同一组 272 道 test 题和 5 个 dev 示例；test 答案不进入提示词。`runs/mmlu_professional_medicine_5shot/` 保存 accuracy、正确题数、逐题配对结果及设置。报告应标明 `professional_medicine` 和 5-shot 协议；它不是完整 MMLU 总分，也不是临床安全评估。公开题目可能出现在预训练数据中。

## 仓库里有什么
当前课程交付物和待办见[提交清单](docs/SUBMISSION_CHECKLIST.md)。Overleaf 源文件保存在本机的 `docs/final-report/`，该目录已加入 Git 忽略；已执行的课程 notebook 在根目录 `Final_Project.ipynb`。Overleaf ZIP 只是报告源文件包，不是完整的课程提交压缩包。


- data/：原始来源和处理后的数据，不纳入 Git。
- models/：下载的模型缓存，不纳入 Git。
- runs/：检查点、TensorBoard 日志、adapter 和运行记录，不纳入 Git。
- docs/research/：中英文研究概览和研究笔记。

Python 代码按下面的职责阅读，所有文件都在 `src/healthcpt/`：

| 步骤 | 文件 |
|---|---|
| 命令入口 | `cli.py` |
| 数据下载与清理 | `medquad.py`、`medical_data.py` |
| 模型训练 | `cpt.py`、`sft.py`、`lstm_baseline.py` |
| 权重合并与文件校验 | `export_hf.py`、`checkpoint_files.py` |
| Qwen 检查与评测 | `verify_hf.py`、`evaluate_qa.py`、`evaluate_mmlu.py` |
| 两类模型共用的问答采样与指标 | `qa_metrics.py` |

CPT、SFT 和 LSTM 基线均已训练。EOS 修正后的 Base/CPT+SFT QA 指标在 `qa_eval_cpt_sft_test_200_eosfix_wsl`，运行环境为 RTX 2080 Ti；早期 RTX 4090 Infra 及 CPT-only 结果保留作历史阶段分析。LSTM BERTScore 和 MMLU professional_medicine 完整科目结果也在 `runs/`。QA 只评测了 1,573 题中的 200 题，人工答案审查仍未完成。部分 CPT+SFT 回答在 EOS 修正后仍重复并达到生成上限。报告已按提交稿修订并在本地编译为 PDF，另有课程交付项待完成。详见[研究概览](docs/research/README.md)、[提交清单](docs/SUBMISSION_CHECKLIST.md)。

## 15. 一键展示最终评测结果

将最终评测和训练记录下载到 `runs/` 后，在项目目录运行：

    python scripts/build_results_dashboard.py --open

命令生成 `runs/final_results.html` 并用浏览器打开。页面可离线查看，支持中英对照、中文、英文和打印。Qwen 问答使用 EOS 修正版 `qa_eval_cpt_sft_test_200_eosfix_wsl`，另展示完整 MMLU 医学子集和最终 LSTM 结果，并读取 CPT/SFT、LSTM 训练记录；不需要模型权重或额外 Python 包。LSTM 来自独立运行，已核对同一批题目和参考答案，但生成长度预算不同；旧缓存 Qwen 分数不用于本页。源文件更新后重新运行即可刷新，页面列出来源路径和 SHA-256。

CPT、SFT 的 loss/accuracy 曲线读取 TensorBoard 训练采样点，缓存为 `runs/training_curves.json`；LSTM 曲线读取逐轮历史。首次生成缓存，或下载新的 TensorBoard 日志后，运行：

    uv run --no-project --with tensorboard python scripts/extract_training_curves.py

然后运行上面的展示页命令。提取过程使用独立 uv 环境中的 TensorBoard，不安装 TensorFlow，也不改变训练环境。

补算 LSTM BERTScore 时，在最终 Qwen Infra 评测使用的 PyTorch 环境运行，无需重新生成回答：

    python src/healthcpt/score_lstm_bertscore.py --dry-run
    python src/healthcpt/score_lstm_bertscore.py

脚本核对已有 200 条回答与最终 Qwen 的题目、参考答案，并要求 BERTScore、Transformers 版本一致。新结果存入 `runs/qa_eval_lstm_16epoch_test_200_bertscore/`，包含 `metrics.json` 和 `predictions.jsonl`，原结果保持不变。下载这个目录到本地，再运行展示页脚本即可显示新增分数。评分只需要 RoBERTa，不加载 Qwen 或 LSTM 权重。

## 16. 课程提交 Notebook

仓库根目录的 `Final_Project.ipynb` 是代码与实验结果的提交入口，已保存执行后的表格与图。内容包括数据准备、真实实现片段、训练和评测入口、指标重算及来源核对，说明采用简短英文。它与源码模块、最终 Word 报告一起使用。

若要重新执行轻量分析，在 Python 3.11 或 3.12 环境安装：

    python -m pip install -r requirements-notebook.txt

在 notebook 编辑器中选择该 Python 内核，再 Run All。需要本地 `data/processed/` 数据，以及 `runs/` 中已下载的评测、训练记录、`training_curves.json` 和 LSTM 词表。默认只读取文件、重算指标和画图，不训练、不推理、不下载模型，也不需要 TensorFlow/PyTorch。没有这些文件时，仍可直接阅读 notebook 已保存的输出。

`RUN_DATA_PREPARATION`、`RUN_TRAINING`、`RUN_EXPORT`、`RUN_INFERENCE`、`RUN_BERTSCORE` 均默认关闭。重跑前按上文准备对应环境，再开启所需操作；输出写入 `runs/notebook_reproduction/`。提交 notebook 时同时附上源码、环境文件及必要结果文件，模型权重可以另行保存。执行 notebook 不代表已完成人工答案评审、专门的可解释性分析或团队最终解释。
