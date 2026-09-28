"""Run a small Transformers text/image check after exporting on the server."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--image", required=True, type=Path, help="A local JPG or PNG")
    args = parser.parse_args()

    # These optional dependencies run separately from the TensorFlow training env.
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
        dtype="auto",
        output_loading_info=True,
    )
    for category in (
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
    ):
        if loading.get(category):
            raise RuntimeError(
                f"Checkpoint loading problem ({category}): {loading[category]}"
            )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(
        str(args.model_dir), local_files_only=True
    )

    # Base models accept continuation prompts; no chat template is required here.
    with Image.open(args.image) as source:
        picture = source.convert("RGB")
    picture.thumbnail((256, 256))
    cases = [
        ("text", "A common symptom of seasonal allergies is", None),
        (
            "image",
            "<|vision_start|><|image_pad|><|vision_end|>\nThis image shows",
            picture,
        ),
    ]
    results = {}
    for name, prompt, image in cases:
        batch = processor(text=[prompt], images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model(**batch, use_cache=False)
            if not torch.isfinite(output.logits).all().item():
                raise RuntimeError(f"Non-finite logits for the {name} input.")
            del output
            tokens = model.generate(
                **batch, max_new_tokens=16, do_sample=False, use_cache=False
            )
        generated = tokens[:, batch["input_ids"].shape[1] :]
        results[name] = processor.batch_decode(generated, skip_special_tokens=True)[0]
        print(f"[Verify] {name}: {results[name]}", flush=True)

    report = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "device": device,
        "image": str(args.image.resolve()),
        "text_and_image_forward_passed": True,
        "generated_text": results,
        "quality_evaluation": "Not performed; this is a loading/inference check only.",
    }
    (args.model_dir / "inference_check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "[Verify] Text and image checks passed. See inference_check.json.", flush=True
    )


if __name__ == "__main__":
    main()
