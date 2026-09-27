"""Bounded TensorFlow/KerasHub continued-pretraining pilot."""

from datetime import datetime, timezone
from pathlib import Path
import json
import os


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
    print(
        "[CPT] Data ready: "
        f"train_examples={len(train_texts)}, "
        f"validation_examples={len(validation_texts)}, "
        f"train_steps_per_epoch={train_steps}, "
        f"validation_steps={validation_steps}, "
        f"batch_size={batch_size}, epochs={epochs}",
        flush=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(5565)
    keras.config.set_dtype_policy("mixed_float16")
    memory_device = "GPU:0"
    tf.config.experimental.reset_memory_stats(memory_device)
    print(f"[CPT] Loading model preset: {preset}", flush=True)
    preprocessor = keras_hub.models.CausalLMPreprocessor.from_preset(
        preset, sequence_length=sequence_length
    )
    if preset.startswith("qwen3_5_"):
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
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
    if preset.startswith("qwen3_5_"):
        model.compile(optimizer=optimizer, jit_compile=False)
    else:
        model.compile(optimizer=optimizer)

    training = tf.data.Dataset.from_tensor_slices(train_texts).shuffle(
        len(train_texts), seed=5565
    ).batch(batch_size)
    validation = tf.data.Dataset.from_tensor_slices(validation_texts).batch(batch_size)
    print(
        "[CPT] Training started. Progress shows steps, ETA, and loss; val_loss appears after validation.",
        flush=True,
    )
    history = model.fit(
        training,
        validation_data=validation,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=validation_steps,
        shuffle=False,
        verbose=1,
    )
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
        "seed": 5565,
        "gpu_allocator_current_bytes": int(gpu_memory["current"]),
        "gpu_allocator_peak_bytes": int(gpu_memory["peak"]),
        "history": {name: [float(x) for x in values] for name, values in history.history.items()},
    }
    (output_dir / "run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
