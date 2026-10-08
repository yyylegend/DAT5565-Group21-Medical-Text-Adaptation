"""Export Qwen TensorBoard training points for the offline dashboard.

Run: uv run --no-project --with tensorboard python scripts/extract_training_curves.py
"""

import hashlib
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.util.tensor_util import make_ndarray

ROOT = Path(__file__).resolve().parents[1]


def main():
    curves = {}
    for stage in ("cpt", "sft"):
        folder = ROOT / f"runs/qwen3_5_2b_{stage}_full_1epoch/tensorboard/train"
        events = EventAccumulator(str(folder), size_guidance={"tensors": 0}).Reload()
        curves[stage] = {
            metric: [[event.step, float(make_ndarray(event.tensor_proto))]
                     for event in events.Tensors(tag)]
            for metric, tag in (("loss", "batch_loss"), ("accuracy", "batch_sparse_categorical_accuracy"))
        }
        curves[stage]["sources"] = [
            {"path": file.relative_to(ROOT).as_posix(),
             "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
            for file in sorted(folder.glob("events.*"))
        ]
        print(f"{stage.upper()}: {len(curves[stage]['loss'])} training points")
    output = ROOT / "runs/training_curves.json"
    output.write_text(json.dumps(curves, ensure_ascii=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
