"""Export a completed Qwen3.5 LoRA run to Hugging Face Safetensors format."""

import json
import os
from pathlib import Path


def export_hf(run_dir: Path, output_dir: Path | None = None) -> dict:
    """Reload the trained adapter and export a standalone text model."""
    run_dir = run_dir.resolve()
    config_path = run_dir / "training_config.json"
    adapter_path = run_dir / "cpt_adapter.lora.h5"
    result_path = run_dir / "run.json"

    # The training script writes run.json only after training finishes.
    if not result_path.is_file():
        raise FileNotFoundError(
            f"{result_path} is missing. Export after CPT has completed successfully."
        )
    if not config_path.is_file() or not adapter_path.is_file():
        raise FileNotFoundError(
            f"Expected training_config.json and cpt_adapter.lora.h5 in {run_dir}."
        )

    # Reuse the model settings saved by the training script.
    training_config = json.loads(config_path.read_text(encoding="utf-8"))
    preset = training_config["preset"]
    lora_rank = int(training_config["lora_rank"])
    sequence_length = int(training_config["sequence_length"])
    if "qwen3_5" not in preset.lower() and "qwen3.5" not in preset.lower():
        raise ValueError(f"This exporter currently supports Qwen3.5, not {preset!r}.")
    if lora_rank < 1:
        raise ValueError("The completed run does not use a LoRA adapter.")

    # Use an empty output folder so an earlier export is not overwritten.
    output_dir = (output_dir or run_dir / "hf_export").resolve()
    if output_dir == run_dir:
        raise ValueError("Choose an export directory separate from the run directory.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Export directory is not empty: {output_dir}. Choose an empty directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set the TensorFlow backend before importing Keras.
    os.environ["KERAS_BACKEND"] = "tensorflow"
    import tensorflow as tf

    # Let TensorFlow use GPU memory as it needs it.
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    import keras
    import keras_hub

    # Match the precision policy used during training.
    keras.config.set_dtype_policy("mixed_float16")

    # This KerasHub exporter writes Transformers-compatible weights and
    # tokenizer files.
    from keras_hub.src.utils.transformers.export.hf_exporter import (
        export_to_safetensors,
    )

    # Rebuild the same text-only Qwen3.5 model used by CPT.
    preprocessor = keras_hub.models.CausalLMPreprocessor.from_preset(
        preset, sequence_length=sequence_length
    )
    backbone = keras_hub.models.Qwen3_5Backbone.from_preset(
        preset, vision_encoder=None
    )
    model = keras_hub.models.Qwen3_5CausalLM(
        backbone=backbone, preprocessor=preprocessor
    )

    # Load the learned LoRA weights. The exporter combines them with the base
    # weights.
    model.backbone.enable_lora(rank=lora_rank)
    model.backbone.load_lora_weights(str(adapter_path))

    print(f"[Export] Writing merged model and tokenizer to {output_dir}", flush=True)
    export_to_safetensors(model, str(output_dir))

    # Check that the main Hugging Face files were created.
    required_files = ("config.json", "model.safetensors", "tokenizer_config.json")
    missing_files = [name for name in required_files if not (output_dir / name).is_file()]
    if missing_files:
        raise RuntimeError(
            f"Export finished without expected file(s): {', '.join(missing_files)}"
        )

    return {
        "export_dir": str(output_dir),
        "format": "Hugging Face Transformers Safetensors",
        "base_preset": preset,
        "lora_rank": lora_rank,
        "vision_encoder": "omitted; text-only export",
        "files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
