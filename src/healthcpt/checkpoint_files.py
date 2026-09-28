"""Copy a Safetensors checkpoint and replace only explicitly selected tensors."""

import hashlib
import json
import shutil
import struct
from pathlib import Path


def checkpoint_layout(base_dir: Path) -> dict:
    """Read tensor locations without loading the model into memory."""
    from safetensors import safe_open

    index_path = base_dir / "model.safetensors.index.json"
    weight_map = None
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        filenames = sorted(set(weight_map.values()))
    else:
        filenames = ["model.safetensors"]

    tensors = {}
    for filename in filenames:
        # Shard names must refer to files inside the supplied checkpoint folder.
        if Path(filename).name != filename:
            raise ValueError(f"Unexpected shard path: {filename}")
        path = base_dir / filename
        with safe_open(str(path), framework="np"):
            pass  # Let Safetensors validate the file before trusting its offsets.
        with path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"Duplicate tensor: {name}")
            start, end = entry["data_offsets"]
            tensors[name] = {
                "file": filename,
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "start": 8 + header_size + start,
                "end": 8 + header_size + end,
            }
    if weight_map is not None:
        actual_map = {name: entry["file"] for name, entry in tensors.items()}
        if actual_map != weight_map:
            raise ValueError("The shard index does not match the stored tensors.")
    return tensors


def _hash_ranges(path: Path, ranges: list[tuple[int, int]]) -> str:
    """Hash selected byte ranges using small buffers, even for large shards."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for start, end in ranges:
            handle.seek(start)
            remaining = end - start
            while remaining:
                block = handle.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError(f"Unexpected end of checkpoint: {path}")
                digest.update(block)
                remaining -= len(block)
    return digest.hexdigest()


def copy_and_patch(
    base_dir: Path, output_dir: Path, tensors: dict, updates: dict
) -> dict:
    """Copy original files, patch text LoRA tensors, and verify all other bytes."""
    if not updates:
        raise ValueError("No LoRA tensor updates were supplied.")
    for name, update in updates.items():
        if not name.startswith("model.language_model."):
            raise ValueError(f"Refusing to replace a non-text tensor: {name}")
        if name not in tensors:
            raise ValueError(f"Updated tensor is absent from the base: {name}")
        entry = tensors[name]
        if update["shape"] != entry["shape"] or update["dtype"] != entry["dtype"]:
            raise ValueError(f"Shape or dtype mismatch: {name}")
        if len(update["data"]) != entry["end"] - entry["start"]:
            raise ValueError(f"Byte length mismatch: {name}")
    vision_count = sum(name.startswith("model.visual.") for name in tensors)
    if not vision_count:
        raise ValueError("The base checkpoint has no model.visual weights.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep the original dtype, tensor names, shard index, and tensor offsets.
    # Equal-sized replacements let us copy BF16 weights without converting them.
    verification = {}
    for filename in sorted({entry["file"] for entry in tensors.values()}):
        source = base_dir / filename
        destination = output_dir / filename
        print(f"[Export] Copying and checking {filename}", flush=True)
        shutil.copyfile(source, destination)
        changed_names = sorted(
            (name for name in updates if tensors[name]["file"] == filename),
            key=lambda name: tensors[name]["start"],
        )
        untouched_ranges = []
        cursor = 0
        with destination.open("r+b") as handle:
            for name in changed_names:
                entry = tensors[name]
                untouched_ranges.append((cursor, entry["start"]))
                handle.seek(entry["start"])
                handle.write(updates[name]["data"])
                cursor = entry["end"]
        untouched_ranges.append((cursor, source.stat().st_size))

        # This checks every untouched byte, including all vision weights.
        expected = _hash_ranges(source, untouched_ranges)
        if _hash_ranges(destination, untouched_ranges) != expected:
            raise RuntimeError(f"Untouched checkpoint bytes changed: {filename}")
        for name in changed_names:
            entry = tensors[name]
            saved_hash = _hash_ranges(destination, [(entry["start"], entry["end"])])
            if saved_hash != hashlib.sha256(updates[name]["data"]).hexdigest():
                raise RuntimeError(
                    f"Saved tensor does not match merged weights: {name}"
                )
        verification[filename] = {"unchanged_bytes_sha256": expected}

    # Preserve the full multimodal config, image/video processor, and tokenizer.
    # A new model card is written by the caller instead of copying the Base card.
    for path in base_dir.iterdir():
        if path.is_file() and (
            path.suffix in {".json", ".txt", ".jinja", ".model"}
            or path.name in {".gitattributes", "LICENSE", "NOTICE"}
        ):
            shutil.copyfile(path, output_dir / path.name)
    if checkpoint_layout(output_dir) != tensors:
        raise RuntimeError(
            "Exported tensor layout differs from the original checkpoint."
        )
    return {"vision_tensors_preserved": vision_count, "shards": verification}
