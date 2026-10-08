"""Continue CPT with supervised question-and-answer examples."""

import hashlib
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    """Return a file hash so a run records exactly which data it used."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _qa_texts(path: Path, limit: int) -> list[str]:
    """Turn each question and answer into one causal language-model example."""
    texts = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not question or not answer:
                raise ValueError(
                    f"Missing question or answer on line {line_number} of {path}"
                )
            texts.append(f"Question: {question}\nAnswer: {answer}")
            if len(texts) >= limit:
                break
    if not texts:
        raise ValueError(f"No complete question-answer examples in {path}")
    return texts


def train(
    cpt_run_dir: Path,
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    limit_train: int,
    limit_validation: int,
    sequence_length: int,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    minimum_learning_rate_ratio: float,
    checkpoint_steps: int,
) -> dict:
    """Fine-tune the saved CPT adapter on question-and-answer examples."""
    cpt_run_dir = cpt_run_dir.resolve()
    cpt_config_path = cpt_run_dir / "training_config.json"
    cpt_run_path = cpt_run_dir / "run.json"
    cpt_adapter_path = cpt_run_dir / "cpt_adapter.lora.h5"
    for path in (cpt_config_path, cpt_run_path, cpt_adapter_path):
        if not path.is_file():
            raise FileNotFoundError(f"Completed CPT file not found: {path}")

    cpt_config = json.loads(cpt_config_path.read_text(encoding="utf-8"))
    cpt_run = json.loads(cpt_run_path.read_text(encoding="utf-8"))
    preset = cpt_config["preset"]
    lora_rank = int(cpt_config["lora_rank"])
    if preset != "hf://Qwen/Qwen3.5-2B-Base":
        raise ValueError("SFT currently supports the project's Qwen3.5-2B Base CPT run.")
    if lora_rank < 1:
        raise ValueError("The CPT run must contain a LoRA adapter.")
    if min(limit_train, limit_validation) < 1:
        raise ValueError("Training and validation limits must be positive.")
    if min(sequence_length, batch_size, checkpoint_steps) < 1:
        raise ValueError("Sequence length, batch size, and checkpoint steps must be positive.")
    if learning_rate <= 0 or not 0 <= warmup_ratio < 1:
        raise ValueError("Learning rate must be positive and warmup ratio must be in [0, 1).")
    if not 0 <= minimum_learning_rate_ratio <= 1:
        raise ValueError("Minimum learning-rate ratio must be between 0 and 1.")

    # Select TensorFlow as Keras's backend before importing Keras.
    os.environ["KERAS_BACKEND"] = "tensorflow"
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("No TensorFlow GPU detected. Run SFT on the Linux GPU server.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    import keras
    import keras_hub

    if keras.backend.backend() != "tensorflow":
        raise RuntimeError("Training backend is not TensorFlow")
    for name, installed in (
        ("tensorflow", tf.__version__),
        ("keras", keras.__version__),
        ("keras_hub", keras_hub.__version__),
    ):
        if installed != cpt_run[name]:
            raise ValueError(
                f"Use the CPT training version {name}=={cpt_run[name]}; found {installed}."
            )

    print("[SFT] Reading question-and-answer data...", flush=True)
    train_texts = _qa_texts(train_path, limit_train)
    validation_texts = _qa_texts(validation_path, limit_validation)
    train_steps = (len(train_texts) + batch_size - 1) // batch_size
    validation_steps = (len(validation_texts) + batch_size - 1) // batch_size
    warmup_steps = min(
        max(1, int(train_steps * warmup_ratio)) if warmup_ratio else 0,
        max(0, train_steps - 1),
    )

    # Record data and settings so an interrupted run can be resumed safely.
    output_dir.mkdir(parents=True, exist_ok=True)
    training_config = {
        "stage": "SFT after CPT",
        "adapter_filename": "sft_adapter.lora.h5",
        "preset": preset,
        "cpt_adapter_sha256": _sha256(cpt_adapter_path),
        "train_sha256": _sha256(train_path),
        "validation_sha256": _sha256(validation_path),
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "epochs": 1,
        "lora_rank": lora_rank,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "minimum_learning_rate_ratio": minimum_learning_rate_ratio,
        "checkpoint_steps": checkpoint_steps,
        "seed": 5565,
    }
    config_path = output_dir / "training_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != training_config:
            raise ValueError(
                f"Data or settings differ from the run in {output_dir}. Use a new output folder."
            )
    else:
        config_path.write_text(json.dumps(training_config, indent=2), encoding="utf-8")

    if (output_dir / "run.json").exists():
        raise FileExistsError(f"This SFT run is already complete: {output_dir}")

    backup_dir = output_dir / "checkpoint"
    metadata_path = backup_dir / "training_metadata.json"
    weights_path = backup_dir / "latest.weights.h5"
    if metadata_path.exists() != weights_path.exists():
        raise RuntimeError(f"Incomplete interruption checkpoint in {backup_dir}")
    resume_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else None
    )
    resume_steps = int(resume_metadata["batch"]) + 1 if resume_metadata else 0
    if resume_steps >= train_steps:
        raise RuntimeError(
            "The checkpoint reached the end of the epoch but the final save is missing. "
            "Keep this folder and contact the project maintainer before restarting."
        )

    # Build the same text-only Qwen backbone used by CPT, then load its adapter.
    keras.utils.set_random_seed(5565)
    keras.config.set_dtype_policy("mixed_float16")
    print(f"[SFT] Loading Qwen3.5-2B and CPT adapter from {cpt_run_dir}", flush=True)
    preprocessor = keras_hub.models.CausalLMPreprocessor.from_preset(
        preset, sequence_length=sequence_length
    )
    backbone = keras_hub.models.Qwen3_5Backbone.from_preset(
        preset, vision_encoder=None
    )
    model = keras_hub.models.Qwen3_5CausalLM(
        backbone=backbone, preprocessor=preprocessor
    )
    model.backbone.enable_lora(rank=lora_rank)
    model.backbone.load_lora_weights(str(cpt_adapter_path))

    total_steps = train_steps
    if warmup_steps:
        learning_rate_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0,
            decay_steps=total_steps - warmup_steps,
            alpha=minimum_learning_rate_ratio,
            warmup_target=learning_rate,
            warmup_steps=warmup_steps,
        )
    else:
        learning_rate_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=total_steps,
            alpha=minimum_learning_rate_ratio,
        )
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate_schedule)
    model.compile(optimizer=optimizer, jit_compile=False)

    random.Random(5565).shuffle(train_texts)
    full_training = tf.data.Dataset.from_tensor_slices(train_texts).batch(batch_size)
    training = full_training.skip(resume_steps) if resume_steps else full_training
    steps_this_run = train_steps - resume_steps
    validation = tf.data.Dataset.from_tensor_slices(validation_texts).batch(batch_size)
    tensorboard_dir = output_dir / "tensorboard"

    class StepAwareBackupAndRestore(keras.callbacks.BackupAndRestore):
        """Save the optimizer and model every few steps for interruption recovery."""

        def on_train_begin(self, logs=None):
            super().on_train_begin(logs)

        def on_train_batch_end(self, batch, logs=None):
            global_batch = int(self.model.optimizer.iterations.numpy()) - 1
            super().on_train_batch_end(global_batch, logs)

    class LearningRateSummary(keras.callbacks.Callback):
        """Write the learning rate beside the other TensorBoard values."""

        def __init__(self):
            super().__init__()
            self.writer = tf.summary.create_file_writer(str(tensorboard_dir / "learning_rate"))

        def on_train_batch_end(self, batch, logs=None):
            step = int(self.model.optimizer.iterations.numpy())
            if step % 100 == 0:
                with self.writer.as_default():
                    tf.summary.scalar("learning_rate", self.model.optimizer.learning_rate, step=step)
                self.writer.flush()

        def on_train_end(self, logs=None):
            self.writer.close()

    print(
        "[SFT] Run setup\n"
        f"  Training examples: {len(train_texts):,}\n"
        f"  Validation examples: {len(validation_texts):,}\n"
        f"  Sequence length: {sequence_length}; batch size: {batch_size}\n"
        f"  Epochs: 1; LoRA rank: {lora_rank}; learning rate: {learning_rate:.1e}\n"
        f"  Steps: {train_steps:,}; warmup: {warmup_steps:,}; checkpoints: every {checkpoint_steps:,}\n"
        f"  Output folder: {output_dir}",
        flush=True,
    )
    if resume_steps:
        print(f"[SFT] Resuming after {resume_steps:,} completed steps.", flush=True)

    # Keras displays step, ETA, loss, and token accuracy while fit() runs.
    history = model.fit(
        training,
        validation_data=validation,
        epochs=1,
        steps_per_epoch=steps_this_run,
        validation_steps=validation_steps,
        shuffle=False,
        verbose=1,
        callbacks=[
            keras.callbacks.TensorBoard(
                log_dir=str(tensorboard_dir),
                update_freq=100,
                write_steps_per_second=True,
            ),
            LearningRateSummary(),
            StepAwareBackupAndRestore(
                backup_dir=str(backup_dir),
                save_freq=checkpoint_steps,
                delete_checkpoint=False,
            ),
        ],
    )

    # Save the combined CPT+SFT adapter. The HF exporter merges it into Base.
    adapter_path = output_dir / "sft_adapter.lora.h5"
    model.backbone.save_lora_weights(str(adapter_path))
    result = {
        "stage": "CPT+SFT",
        "parent_cpt_adapter_sha256": _sha256(cpt_adapter_path),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "preset": preset,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "keras_hub": keras_hub.__version__,
        "gpu": [gpu.name for gpu in gpus],
        "precision": "mixed_float16",
        "train_examples": len(train_texts),
        "validation_examples": len(validation_texts),
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "epochs": 1,
        "lora_rank": lora_rank,
        "learning_rate": learning_rate,
        "learning_rate_schedule": "linear_warmup_cosine_decay",
        "warmup_steps": warmup_steps,
        "minimum_learning_rate_ratio": minimum_learning_rate_ratio,
        "total_training_steps": total_steps,
        "checkpoint_steps": checkpoint_steps,
        "resumed_from_step": resume_steps,
        "seed": 5565,
        "history": {
            name: [float(value) for value in values]
            for name, values in history.history.items()
        },
    }
    (output_dir / "run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    print(
        "[SFT] Training complete.\n"
        f"  Adapter: {adapter_path}\n"
        f"  Run summary: {output_dir / 'run.json'}\n"
        "  Next: run `healthcpt export-hf` to create the full multimodal model.",
        flush=True,
    )
    return result
