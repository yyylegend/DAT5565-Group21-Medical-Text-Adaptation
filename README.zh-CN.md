# DAT5565 第 21 组：医学文本适配

**语言 / Language:** 简体中文 | [English](README.md)

本仓库包含医学问答项目的数据整理脚本和 TensorFlow/Keras CPT（继续预训练）试跑代码。当前 Base → CPT → SFT 方案和数据说明见[研究概览](docs/research/README.md)，也提供[英文版](docs/research/README.en.md)便于对照。训练步骤面向 Linux GPU 服务器，例如租用的云实例。本地 Windows 用户无需启动 WSL 就能检查或传输数据。

## 项目结构

```text
data/raw/                 原始数据和下载文件（不纳入 Git）
data/processed/           清理后的数据和清单（不纳入 Git）
docs/research/            中英文研究概览和执行计划
runs/                     LoRA adapter 和运行记录（不纳入 Git）
src/healthcpt/            数据审计、整理和 CPT 命令
pyproject.toml            项目依赖与 Python 版本
uv.lock                   锁定的依赖版本
```

Python 包只保留当前工作流需要的脚本：

| 文件 | 用途 |
|---|---|
| `src/healthcpt/medquad.py` | 审计 MedQuAD，并按来源划分问答数据。 |
| `src/healthcpt/medical_data.py` | 下载和清理 MedlinePlus/PMC 文本，抽样 CPT 来源，生成文本片段和评测文件。 |
| `src/healthcpt/cpt.py` | 运行 TensorFlow/KerasHub CPT 小规模试跑。 |
| `src/healthcpt/cli.py` | 将项目功能整理为 `healthcpt` 命令。 |

## 训练服务器

- Linux x86_64（Ubuntu 即可）、Python 3.11 或 3.12，以及 `uv`。
- NVIDIA GPU 和可用的驱动；开始前先运行 `nvidia-smi` 确认服务器能看到 GPU。
- 首次安装需要网络，以便获取 Python 依赖和模型权重；也可以预先缓存或使用服务器镜像。

当前锁定环境使用 Python 3.11/3.12、TensorFlow 2.21、Keras 3.15 和 KerasHub 0.32。请选择干净的 Linux GPU 镜像，再用 `uv sync --locked` 安装项目依赖。Python 3.8 和 TensorFlow 1.x 环境不兼容本项目。服务器还需要兼容的 NVIDIA 驱动。具体要求可查 [TensorFlow 安装指南](https://www.tensorflow.org/install/pip)。

请按 [`uv` 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)安装 `uv`。

## 获取项目并检查 GPU

在 Linux GPU 服务器上，把仓库克隆到持久目录中。将占位符替换成持久目录和 GitHub 仓库地址：

```bash
cd YOUR_PERSISTENT_DIRECTORY
git clone YOUR_GROUP_REPOSITORY_URL DAT5565-Final-Project
cd DAT5565-Final-Project
uv sync --locked
```

`uv sync --locked` 会按 `uv.lock` 创建环境并安装依赖。然后检查 TensorFlow 是否能使用 GPU：

```bash
nvidia-smi
uv run python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

第二条命令应该显示至少一个 GPU。如果结果为空，先检查服务器的驱动和 GPU 是否已开放给当前环境，再启动 CPT。

## 准备 MedQuAD

原始压缩包和生成的数据都不纳入 Git。在服务器上下载 [MedQuAD 仓库](https://github.com/abachaa/MedQuAD)的压缩包，或把本地相同的 ZIP 文件传上去，并保存为 `data/raw/medquad-master.zip`：

```bash
mkdir -p data/raw
curl -L https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip \
  -o data/raw/medquad-master.zip
```

研究概览中的统计使用 SHA-256 为 `45aeef400844f3551a7862c3378cc9edf72818ef09d6c1d400f227207ee5179d` 的压缩包。GitHub `master` 内容可能改变；使用新下载的文件前请先检查哈希：

```bash
sha256sum data/raw/medquad-master.zip
```

运行审计，并生成训练/验证/测试问答划分和 CPT 文本：

```bash
uv run healthcpt audit-medquad data/raw/medquad-master.zip
uv run healthcpt prepare-medquad \
  data/raw/medquad-master.zip \
  data/processed/medquad-v1
```

准备脚本会删除缺少问题或答案的记录及完全重复的问答，并按来源网址划分训练、验证和测试集。当前审计得到 16,359 对不同的完整问答，来自 5,486 个至少包含一条完整问答的 XML 文件。v3 CPT 数据将每个文本片段作为一个训练样本：共有 24,240 段，来自 4,401 个来源文档编号。详情见[研究概览](docs/research/README.md)。

## 下载医学文本并准备 CPT 数据

下载最新的 MedlinePlus Health Topic XML，以及从 MedQuAD 常见主题中选出的 PMC 开放获取文章。默认每个主题最多下载 30 篇，覆盖最多 20 个主题（最多 600 篇）；不会下载整个 PMC 数据库。

```bash
uv run healthcpt download-medical-sources \
  data/processed/medquad-v1/qa_train.jsonl \
  data/raw/medical-sources-v3 \
  --topic-limit 20 --articles-per-topic 30
```

原始文件会保存到 `data/raw/medical-sources-v3/medlineplus/` 和 `data/raw/medical-sources-v3/pmc_oa/`。使用带版本号的目录可以保留之前的数据样本。

清理文本、平衡 CPT 来源并生成无明显训练重叠的问答评测文件：

```bash
uv run healthcpt prepare-medical-corpus \
  data/processed/medquad-v1 \
  data/raw/medical-sources-v3 \
  data/processed/cpt-medical-v3 \
  --medquad-cpt-limit 6000 --chunk-words 200
```

这会从 MedQuAD 的训练答案中固定随机抽取 6,000 条用于 CPT，但 SFT 问答训练集保持完整；MedlinePlus 摘要中的 HTML 标签会被清除；长文本会切成最多 200 个空格分词的片段。清单会记录片段数、来源文档数、许可、文件哈希和重叠移除情况。目前 v3 有 24,240 段 CPT 训练文本、1,645 段 CPT 验证文本、1,471 条清理后的 QA 验证样本和 1,573 条清理后的 QA 测试样本。迁移到训练服务器时，请一并传输原始数据、处理后文件和清单；`data/` 已从 Git 排除。

## 运行 Qwen3.5-2B CPT 试跑

先运行一个小规模试跑，确认模型能加载、训练能完成、LoRA adapter 能保存：

```bash
uv run healthcpt cpt-pilot \
  data/processed/cpt-medical-v3/cpt_train.jsonl \
  data/processed/cpt-medical-v3/cpt_validation.jsonl \
  runs/qwen3_5_2b_cpt_pilot \
  --preset qwen3_5_2b_base \
  --limit-train 32 --limit-validation 8 \
  --sequence-length 512 --batch-size 1 --epochs 1 --lora-rank 8
```

KerasHub 0.32 提供 `qwen3_5_2b_base` preset 和 Qwen3.5 模型类，但本项目还没有在目标服务器上成功运行过这条路径。先把它当作试跑：确认模型加载、LoRA 作用层、显存占用和 adapter 保存后，再增加样本量。SFT 训练命令还没有实现。

如果显存不足，可先降低序列长度或 batch size。训练完成后，脚本会把 LoRA adapter 和 `run.json` 写入指定目录。当前没有中途恢复训练的 checkpoint；云服务器若使用临时存储，请选持久化磁盘，并及时备份结果。

`data/` 和 `runs/` 不纳入 Git。第一次 CPT 试跑至少要把 `data/processed/cpt-medical-v3/cpt_train.jsonl` 和 `cpt_validation.jsonl` 传到项目目录下对应的位置。原始压缩包、完整处理数据、模型缓存和运行结果也应保存在持久盘，或切换机器时单独传输。不要把模型权重或数据集提交到代码仓库。

## 当前限制

- 已验证的是旧 Qwen2.5 路径上的两条训练样本试跑；它不能证明 Qwen3.5 兼容、正式训练速度或问答质量会提高。
- v3 CPT 数据已经整理完成，但 Qwen3.5 试跑还没执行；SFT 训练和最终评测代码仍待实现。
- DPO 和 GRPO 是从同一 SFT 检查点分支的可选扩展；仓库中尚未实现。DPO 需要偏好对，GRPO 需要明确奖励信号和兼容的训练框架。
