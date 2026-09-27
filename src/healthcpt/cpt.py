"""TensorFlow/KerasHub continued pretraining with resumable checkpoints."""

import hashlib
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _texts(path: Path, limit: int) -> list[str]:
    texts = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)["text"].strip()
            if value:
                texts.append(value)
            if len(texts) >= limit:
                break
    if not texts:
        raise ValueError(f"No nonempty text in {path}")
    return texts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_qwen3_5_preset(preset: str) -> bool:
    normalized = preset.lower()
    return normalized.startswith("qwen3_5_") or normalized.startswith(
        "hf://qwen/qwen3.5-"
    )


def train(
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    preset: str,
    limit_train: int,
    limit_validation: int,
    sequence_length: int,
    batch_size: int,
    epochs: int,
    lora_rank: int,
    learning_rate: float,
    warmup_ratio: float,
    minimum_learning_rate_ratio: float,
    checkpoint_steps: int,
) -> dict:
    os.environ["KERAS_BACKEND"] = "tensorflow"
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError(
            "No TensorFlow GPU detected. Use a Linux environment with a compatible NVIDIA GPU; "
            "on Windows, use Ubuntu WSL2."
        )
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    import keras
    import keras_hub

    if keras.backend.backend() != "tensorflow":
        raise RuntimeError("Training backend is not TensorFlow")

    print("[CPT] Loading training and validation text...", flush=True)
    train_texts = _texts(train_path, limit_train)
    validation_texts = _texts(validation_path, limit_validation)
    train_steps = (len(train_texts) + batch_size - 1) // batch_size
    validation_steps = (len(validation_texts) + batch_size - 1) // batch_size
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    total_steps = train_steps * epochs
    if checkpoint_steps < 1:
        raise ValueError("checkpoint_steps must be at least 1")
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be between 0 and 1")
    if not 0 <= minimum_learning_rate_ratio <= 1:
        raise ValueError("minimum_learning_rate_ratio must be between 0 and 1")
    warmup_steps = min(
        max(1, int(total_steps * warmup_ratio)) if warmup_ratio else 0,
        max(0, total_steps - 1),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    training_config = {
        "preset": preset,
        "train_sha256": _sha256(train_path),
        "validation_sha256": _sha256(validation_path),
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "epochs": epochs,
        "lora_rank": lora_rank,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "minimum_learning_rate_ratio": minimum_learning_rate_ratio,
        "checkpoint_steps": checkpoint_steps,
        "seed": 5565,
    }
    config_path = output_dir / "training_config.json"
    if config_path.exists():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        if previous_config != training_config:
            raise ValueError(
                f"Training settings or dataset differ from the existing run in {output_dir}. "
                "Use a new output directory instead of resuming it."
            )
    else:
        config_path.write_text(json.dumps(training_config, indent=2), encoding="utf-8")

    backup_dir = output_dir / "checkpoint"
    metadata_path = backup_dir / "training_metadata.json"
    weights_path = backup_dir / "latest.weights.h5"
    resume_metadata = None
    resume_steps = 0
    if metadata_path.exists() or weights_path.exists():
        if not metadata_path.exists() or not weights_path.exists():
            raise RuntimeError(f"Incomplete interruption checkpoint in {backup_dir}")
        resume_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if epochs == 1 and resume_metadata["epoch"] < epochs:
            resume_steps = int(resume_metadata["batch"]) + 1

    print(
        "[CPT] Data ready: "
        f"train_examples={len(train_texts)}, "
        f"validation_examples={len(validation_texts)}, "
        f"train_steps_per_epoch={train_steps}, "
        f"validation_steps={validation_steps}, "
        f"batch_size={batch_size}, epochs={epochs}, "
        f"total_steps={total_steps}, warmup_steps={warmup_steps}, "
        f"minimum_learning_rate={learning_rate * minimum_learning_rate_ratio:.2e}",
        flush=True,
    )
    if resume_metadata is not None:
        if epochs == 1 and resume_metadata["epoch"] < epochs:
            print(
                f"[CPT] Restoring epoch 1 after batch {resume_metadata['batch'] + 1}; "
                f"skipping {resume_steps} completed batches.",
                flush=True,
            )
        else:
            print(
                f"[CPT] Restoring at the start of epoch {resume_metadata['epoch'] + 1}.",
                flush=True,
            )
    keras.utils.set_random_seed(5565)
    keras.config.set_dtype_policy("mixed_float16")
    memory_device = "GPU:0"
    tf.config.experimental.reset_memory_stats(memory_device)
    print(f"[CPT] Loading model preset: {preset}", flush=True)
    preprocessor = keras_hub.models.CausalLMPreprocessor.from_preset(
        preset, sequence_length=sequence_length
    )
    if _is_qwen3_5_preset(preset):
        backbone = keras_hub.models.Qwen3_5Backbone.from_preset(
            preset, vision_encoder=None
        )
        model = keras_hub.models.Qwen3_5CausalLM(
            backbone=backbone, preprocessor=preprocessor
        )
    else:
        model = keras_hub.models.CausalLM.from_preset(
            preset, preprocessor=preprocessor
        )
    if lora_rank:
        model.backbone.enable_lora(rank=lora_rank)
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
    if _is_qwen3_5_preset(preset):
        model.compile(optimizer=optimizer, jit_compile=False)
    else:
        model.compile(optimizer=optimizer)

    random.Random(5565).shuffle(train_texts)
    full_training = tf.data.Dataset.from_tensor_slices(train_texts).batch(batch_size)
    if resume_metadata is not None and epochs == 1 and resume_metadata["epoch"] < epochs:
        resume_steps = min(int(resume_metadata["batch"]) + 1, train_steps)
    else:
        resume_steps = 0
    resume_epoch = int(resume_metadata["epoch"]) if resume_metadata else 0
    if resume_steps == train_steps and resume_epoch < epochs:
        resume_epoch += 1
    if resume_metadata is None:
        resumed_from_step = 0
    elif epochs == 1:
        resumed_from_step = min(int(resume_metadata["batch"]) + 1, train_steps)
    else:
        resumed_from_step = min(int(resume_metadata["epoch"]) * train_steps, total_steps)
    if resume_epoch < epochs and resume_steps:
        training = full_training.skip(resume_steps)
        steps_this_epoch = train_steps - resume_steps
    else:
        training = full_training
        steps_this_epoch = train_steps
    validation = tf.data.Dataset.from_tensor_slices(validation_texts).batch(batch_size)

    class StepAwareBackupAndRestore(keras.callbacks.BackupAndRestore):
        def on_train_begin(self, logs=None):
            super().on_train_begin(logs)
            self._last_batch_seen = int(self.model.optimizer.iterations.numpy()) - 1

        def on_train_batch_end(self, batch, logs=None):
            global_batch = int(self.model.optimizer.iterations.numpy()) - 1
            super().on_train_batch_end(global_batch, logs)
            if self.save_freq != "epoch" and (global_batch + 1) % checkpoint_steps == 0:
                print(f"[CPT] Checkpoint saved at optimizer step {global_batch + 1}.", flush=True)

        def on_epoch_end(self, epoch, logs=None):
            super().on_epoch_end(epoch, logs)
            self._last_batch_seen = int(self.model.optimizer.iterations.numpy()) - 1
            if self.save_freq == "epoch":
                print(f"[CPT] Checkpoint saved after epoch {epoch + 1}.", flush=True)

    class LearningRateSummary(keras.callbacks.Callback):
        def __init__(self, log_dir: Path):
            super().__init__()
            self.writer = tf.summary.create_file_writer(str(log_dir))

        def on_train_batch_end(self, batch, logs=None):
            step = int(self.model.optimizer.iterations.numpy())
            if step % 100 == 0:
                with self.writer.as_default():
                    tf.summary.scalar(
                        "learning_rate", self.model.optimizer.learning_rate, step=step
                    )
                self.writer.flush()

        def on_train_end(self, logs=None):
            self.writer.close()

    print(
        "[CPT] Training started. Progress shows steps, ETA, and loss; val_loss appears after validation.",
        flush=True,
    )
    tensorboard_log_dir = output_dir / "tensorboard"
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=str(tensorboard_log_dir),
        update_freq=100,
        write_steps_per_second=True,
    )
    print(
        f"[CPT] TensorBoard scalars will be written to {tensorboard_log_dir} every 100 steps.",
        flush=True,
    )
    checkpoint_frequency = checkpoint_steps if epochs == 1 else "epoch"
    print(
        f"[CPT] Interruption checkpoints: {checkpoint_frequency} in {backup_dir}. "
        "Rerun the same command to resume from the latest checkpoint.",
        flush=True,
    )
    if resume_steps:
        print(
            f"[CPT] Resuming after {resume_steps} completed steps; "
            f"{steps_this_epoch} training steps remain in this epoch.",
            flush=True,
        )
    history = model.fit(
        training,
        validation_data=validation,
        epochs=epochs,
        initial_epoch=resume_epoch,
        steps_per_epoch=steps_this_epoch,
        validation_steps=validation_steps,
        shuffle=False,
        verbose=1,
        callbacks=[
            tensorboard_callback,
            LearningRateSummary(tensorboard_log_dir / "learning_rate"),
            StepAwareBackupAndRestore(
                backup_dir=str(backup_dir),
                save_freq=checkpoint_frequency,
                delete_checkpoint=False,
            ),
        ],
    )
    if not history.history:
        validation_metrics = model.evaluate(
            validation, steps=validation_steps, verbose=1, return_dict=True
        )
        history.history = {
            f"val_{name}": [float(value)] for name, value in validation_metrics.items()
        }
    gpu_memory = tf.config.experimental.get_memory_info(memory_device)

    if lora_rank:
        model.backbone.save_lora_weights(str(output_dir / "cpt_adapter.lora.h5"))
    else:
        model.save_to_preset(str(output_dir / "cpt_preset"))
    result = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "preset": preset,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "keras_hub": keras_hub.__version__,
        "gpu": [gpu.name for gpu in gpus],
        "precision": "mixed_float16",
        "train_documents": len(train_texts),
        "validation_documents": len(validation_texts),
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "epochs": epochs,
        "lora_rank": lora_rank,
        "learning_rate": learning_rate,
        "learning_rate_schedule": "linear_warmup_cosine_decay",
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "minimum_learning_rate_ratio": minimum_learning_rate_ratio,
        "total_training_steps": total_steps,
        "checkpoint_steps": checkpoint_steps,
        "resumed_from_step": resumed_from_step,
        "seed": 5565,
        "gpu_allocator_current_bytes": int(gpu_memory["current"]),
        "gpu_allocator_peak_bytes": int(gpu_memory["peak"]),
        "history": {name: [float(x) for x in values] for name, values in history.history.items()},
    }
    (output_dir / "run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    return result
