"""Merge text LoRA updates into the original full Qwen3.5 checkpoint."""

import hashlib
import json
import os
from pathlib import Path

from .checkpoint_files import checkpoint_layout, copy_and_patch


def merged_projection(layer, hidden_dim: int):
    """Add LoRA in float32 and undo the HF-to-Keras transpose/reshape."""
    import numpy as np

    # Read stored variables rather than the fp16 compute-time kernel property.
    kernel = layer._kernel.numpy().astype(np.float32)
    a = layer.lora_kernel_a.numpy().astype(np.float32)
    b = layer.lora_kernel_b.numpy().astype(np.float32)
    scale = layer.lora_alpha / layer.lora_rank
    merged = kernel + scale * np.matmul(a, b)
    return merged.reshape(hidden_dim, -1).T.copy()


def encode_tensor(values, entry: dict) -> dict:
    """Save the merged tensor in the dtype and shape of its original HF tensor."""
    import ml_dtypes
    import numpy as np

    dtypes = {"BF16": ml_dtypes.bfloat16, "F16": np.float16, "F32": np.float32}
    if list(values.shape) != entry["shape"] or entry["dtype"] not in dtypes:
        raise ValueError("Merged tensor shape or storage dtype is unsupported.")
    values = values.astype(dtypes[entry["dtype"]])
    if not np.isfinite(values).all():
        raise ValueError("Merged tensor contains NaN or infinity.")
    return {
        "shape": list(values.shape),
        "dtype": entry["dtype"],
        "data": values.tobytes(),
    }


def export_hf(run_dir: Path, base_dir: Path, output_dir: Path | None = None) -> dict:
    """Merge a trained text adapter while preserving the original vision weights."""
    run_dir, base_dir = run_dir.resolve(), base_dir.resolve()
    if not (run_dir / "training_config.json").is_file():
        raise FileNotFoundError(f"Training configuration not found in {run_dir}")
    settings = json.loads(
        (run_dir / "training_config.json").read_text(encoding="utf-8")
    )
    adapter_name = settings.get("adapter_filename", "cpt_adapter.lora.h5")
    for name in ("run.json", "training_config.json", adapter_name):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(
                f"A completed training run must contain {run_dir / name}"
            )
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if settings["lora_rank"] < 1:
        raise ValueError("This exporter requires a LoRA training run.")
    if settings["preset"] != "hf://Qwen/Qwen3.5-2B-Base":
        raise ValueError(
            "This merge path requires a run trained from hf://Qwen/Qwen3.5-2B-Base."
        )

    # Use the local snapshot used for training. Do not fetch today's moving main.
    config = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5" or not config.get("vision_config"):
        raise ValueError(
            "--base-dir must contain the original full multimodal Base model."
        )
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "LICENSE",
    ):
        if not (base_dir / name).is_file():
            raise FileNotFoundError(f"Original model asset missing: {base_dir / name}")
    tensors = checkpoint_layout(base_dir)
    output_dir = (output_dir or run_dir / "hf_export_multimodal").resolve()
    if output_dir == base_dir or base_dir in output_dir.parents:
        raise ValueError("Write the export outside the original Base snapshot.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty output folder: {output_dir}")

    os.environ["KERAS_BACKEND"] = "tensorflow"
    import keras
    import keras_hub
    import tensorflow as tf

    # KerasHub stores this adapter's layers by index. Match the training versions.
    for name, installed in (
        ("tensorflow", tf.__version__),
        ("keras", keras.__version__),
        ("keras_hub", keras_hub.__version__),
    ):
        if installed != run[name]:
            raise ValueError(
                f"Use training version {name}=={run[name]}; found {installed}."
            )
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    keras.config.set_dtype_policy("mixed_float16")

    # Rebuild the same text backbone so the saved adapter layer indices match.
    print(f"[Export] Loading original Base snapshot: {base_dir}", flush=True)
    backbone = keras_hub.models.Qwen3_5Backbone.from_preset(
        str(base_dir), vision_encoder=None
    )
    backbone.enable_lora(rank=settings["lora_rank"])
    backbone.load_lora_weights(str(run_dir / adapter_name))

    updates = {}
    mapped_layers = set()
    for index, decoder in enumerate(backbone.transformer_layers):
        if decoder.layer_type != "full_attention":
            continue
        attention = decoder._self_attention_layer
        for projection, layer in (
            ("q_proj", attention._query_dense),
            ("v_proj", attention._value_dense),
        ):
            if not layer.lora_enabled:
                continue
            name = f"model.language_model.layers.{index}.self_attn.{projection}.weight"
            if name not in tensors:
                raise ValueError(f"Text projection missing from original Base: {name}")
            updates[name] = encode_tensor(
                merged_projection(layer, backbone.hidden_dim), tensors[name]
            )
            mapped_layers.add(id(layer))

    # Fail if an adapter target was not covered by our Qwen3.5 weight mapping.
    adapter_layers = {
        id(layer)
        for layer in backbone._flatten_layers()
        if getattr(layer, "lora_enabled", False)
    }
    if mapped_layers != adapter_layers:
        raise ValueError("Some LoRA layers have no checked HF tensor mapping.")
    if len(updates) != 12:
        raise ValueError(
            f"Expected 12 Qwen3.5-2B text projections, found {len(updates)}."
        )
    print(
        f"[Export] Merging {len(updates)} text projections; retaining original vision.",
        flush=True,
    )
    report = copy_and_patch(base_dir, output_dir, tensors, updates)
    report.update(
        {
            "base_preset": settings["preset"],
            "base_snapshot_name": base_dir.name,
            "base_revision_recorded_during_training": False,
            "adapter_file": adapter_name,
            "adapter_sha256": hashlib.sha256((run_dir / adapter_name).read_bytes()).hexdigest(),
            "updated_tensors": sorted(updates),
            "keras": keras.__version__,
            "keras_hub": keras_hub.__version__,
            "multimodal_inference_tested": False,
            "note": "Original vision bytes are preserved. Image/video quality after text adaptation is untested.",
        }
    )
    for name in ("run.json", "training_config.json"):
        (output_dir / name).write_bytes((run_dir / name).read_bytes())
    stage = run.get("stage", "CPT")
    (output_dir / "README.md").write_text(
        "# Qwen3.5-2B medical model\n\n"
        f"Base: Qwen/Qwen3.5-2B-Base. Text LoRA updates were merged for {stage}.\n"
        "Original vision weights, processor, tokenizer, and configuration are preserved.\n"
        "This is a Base model adapted on text; image/video quality has not been evaluated.\n"
        "See export_report.json for the merge checks and training files for settings.\n",
        encoding="utf-8",
    )
    # Write the completion report only after all byte-preservation checks pass.
    (output_dir / "export_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"[Export] Complete: {output_dir}", flush=True)
    return {
        "export_dir": str(output_dir),
        "updated_text_tensors": len(updates),
        "vision_tensors_preserved": report["vision_tensors_preserved"],
        "multimodal_inference_tested": False,
    }
