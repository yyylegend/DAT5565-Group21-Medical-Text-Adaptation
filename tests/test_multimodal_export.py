"""Small checkpoints test merging without downloading a model or using a GPU."""

import json
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from healthcpt.checkpoint_files import checkpoint_layout, copy_and_patch
from healthcpt.export_hf import encode_tensor, merged_projection

QUERY = "model.language_model.layers.3.self_attn.q_proj.weight"
VISION = "model.visual.merger.linear_fc1.weight"
FROZEN = "model.language_model.layers.3.self_attn.k_proj.weight"


def write_checkpoint(path, records):
    # Build tiny real Safetensors files, including BF16 without a torch install.
    header, payload = {}, bytearray()
    for name, dtype, shape, data in records:
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class MultimodalExportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name) / "base"
        self.out = Path(self.temp.name) / "export"
        self.base.mkdir()
        for name in (
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "video_preprocessor_config.json",
            "tokenizer_config.json",
        ):
            (self.base / name).write_text('{"original": true}', encoding="utf-8")
        (self.base / "LICENSE").write_text("original license", encoding="utf-8")
        (self.base / ".gitattributes").write_text(
            "*.safetensors filter=lfs\n", encoding="utf-8"
        )

    def fixture(self, sharded=False):
        records = [
            (QUERY, "BF16", [2, 2], struct.pack("<4H", 0x3F80, 0x4000, 0x4040, 0x4080)),
            (VISION, "BF16", [2], struct.pack("<2H", 0x3F81, 0xBF81)),
            (FROZEN, "F32", [1], struct.pack("<f", 7.25)),
        ]
        if sharded:
            write_checkpoint(self.base / "part-1.safetensors", records[:1])
            write_checkpoint(self.base / "part-2.safetensors", records[1:])
            (self.base / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            QUERY: "part-1.safetensors",
                            VISION: "part-2.safetensors",
                            FROZEN: "part-2.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
        else:
            write_checkpoint(self.base / "model.safetensors", records)
        return checkpoint_layout(self.base)

    def test_preserve_vision_frozen_text_and_assets_for_single_and_sharded_files(self):
        for sharded in (False, True):
            with self.subTest(sharded=sharded):
                tensors = self.fixture(sharded)
                out = self.out / str(sharded)
                original = {p.name: p.read_bytes() for p in self.base.iterdir()}
                update = encode_tensor(
                    np.array([[5, 6], [7, 8]], dtype=np.float32), tensors[QUERY]
                )
                report = copy_and_patch(self.base, out, tensors, {QUERY: update})
                self.assertEqual(report["vision_tensors_preserved"], 1)
                for name, entry in tensors.items():
                    data = (out / entry["file"]).read_bytes()[
                        entry["start"] : entry["end"]
                    ]
                    expected = (
                        update["data"]
                        if name == QUERY
                        else original[entry["file"]][entry["start"] : entry["end"]]
                    )
                    self.assertEqual(data, expected)
                for name, content in original.items():
                    self.assertEqual((self.base / name).read_bytes(), content)
                    if not name.endswith(".safetensors"):
                        self.assertEqual((out / name).read_bytes(), content)
                with safe_open(
                    str(out / tensors[QUERY]["file"]), framework="np"
                ) as handle:
                    self.assertEqual(handle.get_slice(QUERY).get_shape(), [2, 2])

    def test_reject_incompatible_or_visual_updates_before_writing(self):
        tensors = self.fixture()
        update = encode_tensor(np.ones((2, 2)), tensors[QUERY])
        cases = [
            {VISION: update},
            {QUERY + ".missing": update},
            {QUERY: dict(update, shape=[4])},
            {QUERY: dict(update, dtype="F16")},
            {QUERY: dict(update, data=b"short")},
            {},
        ]
        for changes in cases:
            with self.subTest(changes=list(changes)):
                with self.assertRaises(ValueError):
                    copy_and_patch(self.base, self.out, tensors, changes)
                self.assertFalse(self.out.exists())

    def test_reject_incorrect_shard_index(self):
        self.fixture(sharded=True)
        index = self.base / "model.safetensors.index.json"
        data = json.loads(index.read_text())
        del data["weight_map"][VISION]
        index.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            checkpoint_layout(self.base)

    def test_refuse_overwriting_existing_output(self):
        tensors = self.fixture()
        update = encode_tensor(np.ones((2, 2)), tensors[QUERY])
        self.out.mkdir()
        marker = self.out / "existing.txt"
        marker.write_text("keep me")
        with self.assertRaises(FileExistsError):
            copy_and_patch(self.base, self.out, tensors, {QUERY: update})
        self.assertEqual(marker.read_text(), "keep me")

    def test_detect_corruption_of_preserved_vision_bytes(self):
        tensors = self.fixture()
        update = encode_tensor(np.ones((2, 2)), tensors[QUERY])
        original_copy = shutil.copyfile

        def corrupt_copy(source, destination):
            original_copy(source, destination)
            with destination.open("r+b") as handle:
                handle.seek(tensors[VISION]["start"])
                handle.write(b"\x00")

        with (
            patch(
                "healthcpt.checkpoint_files.shutil.copyfile", side_effect=corrupt_copy
            ),
            self.assertRaisesRegex(RuntimeError, "Untouched checkpoint bytes changed"),
        ):
            copy_and_patch(self.base, self.out, tensors, {QUERY: update})

    def test_float32_export_is_readable_by_safetensors(self):
        write_checkpoint(
            self.base / "model.safetensors",
            [
                (QUERY, "F32", [1], struct.pack("<f", 1.0)),
                (VISION, "F32", [1], struct.pack("<f", 2.0)),
            ],
        )
        tensors = checkpoint_layout(self.base)
        update = encode_tensor(np.array([3.0]), tensors[QUERY])
        copy_and_patch(self.base, self.out, tensors, {QUERY: update})
        with safe_open(str(self.out / "model.safetensors"), framework="np") as handle:
            np.testing.assert_array_equal(handle.get_tensor(QUERY), [3.0])
            np.testing.assert_array_equal(handle.get_tensor(VISION), [2.0])

    def test_merge_with_real_keras_einsum_layer(self):
        import keras

        layer = keras.layers.EinsumDense("bi,ihd->bhd", output_shape=(2, 2))
        inputs = np.array([[0.3, -0.7]], dtype=np.float32)
        layer(inputs)
        layer._kernel.assign(np.arange(8, dtype=np.float32).reshape(2, 2, 2))
        layer.enable_lora(rank=1, lora_alpha=2)
        layer.lora_kernel_a.assign(np.ones((2, 2, 1), dtype=np.float32))
        layer.lora_kernel_b.assign(np.array([[0.5, -0.25]], dtype=np.float32))
        expected = np.asarray(layer(inputs)).reshape(1, -1)
        actual = inputs @ merged_projection(layer, hidden_dim=2).T
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_lora_merge_matches_projection_with_nonunit_scale(self):
        kernel = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        a = np.array([[[1], [2]], [[3], [4]]], dtype=np.float32)
        b = np.array([[0.5, -0.25]], dtype=np.float32)
        variable = lambda value: SimpleNamespace(numpy=lambda: value)
        layer = SimpleNamespace(
            _kernel=variable(kernel),
            lora_kernel_a=variable(a),
            lora_kernel_b=variable(b),
            lora_alpha=2,
            lora_rank=1,
        )
        inputs = np.array([[0.3, -0.7]], dtype=np.float32)
        expected = (
            np.einsum("bi,ihd->bhd", inputs, kernel)
            + 2 * np.einsum("bi,ihr,rd->bhd", inputs, a, b)
        ).reshape(1, -1)
        actual = inputs @ merged_projection(layer, hidden_dim=2).T
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_bfloat16_encoding_and_nonfinite_rejection(self):
        entry = {"shape": [2], "dtype": "BF16"}
        self.assertEqual(
            encode_tensor(np.array([1.0, 2.0]), entry)["data"],
            struct.pack("<2H", 0x3F80, 0x4000),
        )
        with self.assertRaises(ValueError):
            encode_tensor(np.array([float("nan"), 1.0]), entry)


if __name__ == "__main__":
    unittest.main()
