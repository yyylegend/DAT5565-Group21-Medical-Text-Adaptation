# Medical Question Answering Model: Research and Data Preparation

Updated: 2026-09-28. The submitted proposal remains unchanged. This document records the current experiment plan, v3 data status, and completed CPT stage. See the [简体中文版](README.md).

## Project question

We selected Qwen3.5-2B-Base for a small medical question-answering prototype. We want to learn whether it can draft answers to common health questions for online healthcare staff to review. If a small model can produce useful drafts at a low operating cost, it may reduce time spent answering repetitive questions. This is a course experiment, not a diagnostic system or a replacement for a clinician.

The main workflow is **Base → CPT → SFT**. Continued pretraining (CPT) lets the model learn from medical text. Supervised fine-tuning (SFT) then teaches it to answer questions. The main comparison is the Base model versus the final CPT+SFT model on the same held-out questions. This measures the overall training pipeline; it cannot isolate how much CPT contributes beyond SFT. That differs from the submitted proposal, which planned to compare CPT+SFT with SFT alone. The final report should explain this change in scope and why it was made.

DPO and GRPO are optional extensions if time, data, and the TensorFlow/Keras toolchain allow. Both would branch from the same SFT checkpoint. DPO needs pairs of preferred and less-preferred answers. GRPO also needs a clear and reliable scoring rule. These datasets, reward methods, and training implementations are not yet defined, so they are not part of the core plan.

## What the datasets contain

| Source | Original data and preparation | Current use |
|---|---|---|
| [MedQuAD](https://github.com/abachaa/MedQuAD) | Each XML source file can contain one or more question-answer pairs. The archive has 11,264 XML files; 5,486 contain at least one complete pair. After cleaning and deduplication, there are **16,359 complete pairs**: 12,799 train, 1,713 validation, and 1,847 test, split by source URL. | SFT still uses all 12,799 training pairs. There are 12,371 distinct training answer texts available for CPT; v3 uses a fixed random sample of 6,000 to reduce the answer-text share in CPT. |
| [MedlinePlus Health Topic XML](https://medlineplus.gov/xml.html) | Each health topic has a title, language, URL, and a plain-language summary. The downloaded version is dated 2026-09-26 and contains 2,033 topics. After excluding 187 pages linked to MedQuAD validation/test records, non-English topics, and topics without summaries, 827 English summaries remain: 813 train and 14 validation. Embedded HTML tags were removed while keeping their visible text. | All are kept for CPT. With a limit of 200 whitespace-separated words per chunk, they produce 1,758 train chunks and 50 validation chunks. |
| [PMC Open Access](https://pmc.ncbi.nlm.nih.gov/tools/textmining/) | Each JATS XML article includes a title, abstract, sectioned body, and references. We searched up to 20 common MedQuAD topics, requesting up to 30 articles per topic and filtering by article-level CC BY/CC0 license metadata. We selected 571 articles; 9 correction, retraction, or similar notice records were removed, leaving 562: 502 train and 60 validation. Titles, abstracts, and body text are extracted; references and tables are skipped. | The article text is split into chunks of up to 200 whitespace-separated words, yielding 13,118 training chunks and 1,595 validation chunks. PMC licenses vary by article; the manifest preserves each license. See [PMC’s official guidance](https://pmc.ncbi.nlm.nih.gov/tools/textmining/). |

The original MedQuAD SFT rows contain a `question`, a reference `answer`, and fields such as topic and source URL. CPT uses plain text as learning material: MedQuAD answer text, MedlinePlus topic title plus summary, or PMC article title, abstract, and body. CPT JSONL rows keep the text, source, document ID, and license metadata. Long documents are split into smaller rows with a `chunk_index`.

Simplified examples: an SFT row is `{"question":"What is asthma?","answer":"Asthma is ..."}`. A CPT row can look like `{"text":"[one chunk from the paper]","source":"PMC","document":"PMC...","chunk_index":3,"license":"CC BY"}`. CPT used a 512-token input length. A 200-word chunk is not necessarily 512 tokenizer tokens, and KerasHub pads or truncates text to a fixed sequence length. We have not separately measured the truncation rate. See the [KerasHub preprocessor documentation](https://keras.io/keras_hub/api/base_classes/causal_lm_preprocessor/).

## v3 CPT mixture

The v3 CPT training set contains **24,240 text chunks**:

- MedQuAD: 6,000 sampled answer texts become 9,364 chunks from 3,086 XML source documents.
- MedlinePlus: 813 health topics become 1,758 chunks.
- PMC: 502 training papers become 13,118 chunks.

The CPT validation set contains 1,645 chunks: 50 MedlinePlus chunks and 1,595 PMC chunks, from 14 MedlinePlus topics and 60 papers. The MedQuAD SFT training split remains complete.

By a rough whitespace-separated word count, the training text totals about **4.07 million words**: about 2.57 million from PMC (63%), 1.22 million from MedQuAD (30%), and 0.28 million from MedlinePlus (7%). These are not model-token counts. The course instructions call for at least 10,000 text documents. We count each prepared CPT chunk as one training example, giving 24,240. We also report the underlying source-document IDs: 4,401 in the training set, so chunk counts are not mistaken for distinct articles.

## Data flow and evaluation

```mermaid
flowchart LR
    M[MedQuAD training QA] --> A[Sample 6,000 unique answers]
    M --> Q[SFT: questions and reference answers]
    P[MedlinePlus summaries] --> C[Medical text]
    R[PMC article text] --> C
    A --> C
    C --> K[Chunks of up to 200 words]
    B[Base model] --> T[CPT]
    K --> T
    T --> S[SFT]
    Q --> S
    B --> E[Same held-out questions]
    S --> E
    S -. optional .-> D[DPO]
    S -. optional .-> G[GRPO]
    D --> E
    G --> E
```

The original MedQuAD splits remain unchanged. Separate cleaned evaluation files contain 1,471 validation questions and 1,573 test questions. They remove questions or answers that exactly repeat SFT training content. The current checks cover exact duplicates and some full-answer matches; they do not include a semantic near-duplicate audit, so they cannot guarantee that all knowledge overlap is gone.

The Base, CPT, and CPT+SFT models use the same questions, prompt, and generation settings. `evaluate_qa.py` uses the `Question: {question}\nAnswer:` prompt with greedy decoding, and reports normalized exact match, token F1, ROUGE-L, and per-question answers for manual review of relevance, missing information, and unsupported claims. Text-overlap metrics do not establish medical correctness; the human-scoring rubric is still to be defined, and there is no clinical validation.

## Current status

- MedQuAD, MedlinePlus, and 571 PMC articles have been downloaded. The v3 CPT corpus has been cleaned, sampled, and chunked.
- The current manifest is `data/processed/cpt-medical-v3/manifest.json`. v1 and v2 remain available. Raw and processed data are not tracked by Git and must be transferred to the training server separately.
- On 2026-09-28, one epoch of Qwen3.5-2B-Base CPT completed with 24,240 training text examples and 1,645 validation examples, sequence length 512, batch size 1, LoRA rank 8, and peak learning rate 1e-4. Training loss was 0.920 and validation loss was 1.229; token accuracy was 0.570 and 0.538, respectively. These are next-token language-model metrics, not QA accuracy.
- The full Hugging Face-format export is at `runs/qwen3_5_2b_cpt_full_1epoch/hf_export_multimodal`. It uses Base snapshot `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`, merges 12 text q/v projection weights, and preserves all 297 vision weights unchanged. The run folder keeps the CPT adapter and `run.json`; the export folder contains `export_report.json` and `inference_check.json`.
- Transformers successfully loaded all 617 weights and completed text and image inference. The image check used a generated solid-color PNG, so it only confirms that the model loads and accepts image input. Real-image capability comparisons and manual QA review remain undone.
- SFT completed one epoch on the server with 12,799 QA training pairs and 1,471 validation pairs, sequence length 512, batch size 1, LoRA rank 8, learning rate 2e-5, and 639 warmup steps. Validation loss was 0.6134 and token accuracy was 0.7025; training loss was 0.5879 and token accuracy was 0.6993. These are next-token metrics on concatenated QA text, not medical-answer accuracy. Long examples may be truncated at 512 tokens; the rate has not been measured.
- The full Hugging Face SFT export is at `runs/qwen3_5_2b_sft_full_1epoch/hf_export_multimodal`. It merges the cumulative CPT+SFT LoRA update into the original Base Safetensors model and preserves the original vision weights.
- `evaluate_qa.py` used the same prompt, greedy decoding, a 128-token limit, and seed 5565 in the following four 200-example evaluations. Each row lists Base and candidate scores from the same paired run:

  | Split | Candidate | Base token F1 | Candidate token F1 | Base ROUGE-L | Candidate ROUGE-L |
  |---|---|---:|---:|---:|---:|
  | Validation, n=200 | CPT | 0.2499 | 0.2345 | 0.1606 | 0.1639 |
  | Validation, n=200 | CPT+SFT | 0.2486 | 0.3421 | 0.1598 | 0.2702 |
  | Test, n=200 | CPT | 0.2540 | 0.2332 | 0.1582 | 0.1638 |
  | Test, n=200 | CPT+SFT | 0.2586 | 0.3248 | 0.1600 | 0.2622 |

  Normalized exact match was 0 for all four model runs, and there were no empty predictions. CPT had lower token F1 and slightly higher ROUGE-L than its paired Base run. CPT+SFT scored higher than its paired Base on both text-overlap metrics in both 200-example samples. These preliminary results do not establish medical correctness. The two test runs used the same file hash, seed, and generation settings, but their Base scores varied slightly; interpret the paired scores within each run rather than treating Base values from different runs as identical. Per-question predictions and JSON summaries are on the server under `runs/qa_eval_*`; they are not tracked by Git.
- The test file contains 1,573 rows; only a seed-5565 sample of 200 has been evaluated so far. The full test split has not been run. MedQuAD is public, but this held-out split comes from the same dataset family as the training data; if time allows, add a small independent consumer-health QA evaluation.
- SFT training and export are on the personal work branch `runqi/sft-work`. The `main` branch remains the CPT baseline for now; decide whether to merge the completed SFT pipeline after results and documentation are finalized.
- Training targets a Linux GPU server with dependencies managed by `uv`. WSL is not needed to prepare or inspect the data locally.

If DPO or GRPO is attempted later, both should branch independently from the same SFT checkpoint. Define and version the preference data, reward rule, and evaluation set before starting. A higher reward score alone does not prove that answers improved. If the toolchain or data are not ready in time, complete the Base→CPT→SFT workflow.

## Recommended reading

- [BioMedLM: A 2.7B Parameter Language Model Trained On Biomedical Text](https://arxiv.org/abs/2403.18421): a reference for the order of medical-text training followed by question-answer fine-tuning. It trains from scratch with far more data and compute, so its scale is not copied here.
- [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://aclanthology.org/2020.acl-main.740/): studies continued training on domain text followed by task adaptation, which helps explain the role of CPT in this project.
