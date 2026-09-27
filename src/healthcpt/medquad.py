"""Audit and prepare MedQuAD without mixing source documents across splits."""

from collections import Counter
from hashlib import sha256
from html import unescape
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile
import json
import re


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value)).strip()


def extract_records(archive: Path):
    with ZipFile(archive) as zf:
        for name in sorted(zf.namelist()):
            if not name.lower().endswith(".xml"):
                continue
            root = ET.fromstring(zf.read(name))
            source = root.attrib.get("source") or name.split("/")[1]
            url = root.attrib.get("url", "")
            group = url or f"{source}/{root.attrib.get('id', name)}"
            topic = clean_text(root.findtext("Focus") or "")
            for pair in root.findall(".//QAPair"):
                question_node = pair.find("Question")
                answer_node = pair.find("Answer")
                question = clean_text("".join(question_node.itertext())) if question_node is not None else ""
                answer = clean_text("".join(answer_node.itertext())) if answer_node is not None else ""
                yield {
                    "document": name,
                    "group": group,
                    "source": source,
                    "topic": topic,
                    "question_type": question_node.attrib.get("qtype", "") if question_node is not None else "",
                    "question": question,
                    "answer": answer,
                    "url": url,
                }


def split_name(group: str, seed: int) -> str:
    number = int.from_bytes(sha256(f"{seed}:{group}".encode()).digest()[:8], "big") / 2**64
    return "train" if number < 0.8 else "validation" if number < 0.9 else "test"


def audit(archive: Path) -> dict:
    sources = Counter()
    usable_documents = set()
    missing = Counter()
    unique_pairs = set()
    answer_hashes = set()
    for row in extract_records(archive):
        sources[row["source"]] += 1
        if not row["question"] or not row["answer"]:
            missing[row["source"]] += 1
            continue
        usable_documents.add(row["document"])
        unique_pairs.add((row["question"].casefold(), row["answer"].casefold()))
        answer_hashes.add(sha256(row["answer"].casefold().encode()).hexdigest())
    return {
        "xml_documents": len({row["document"] for row in extract_records(archive)}),
        "usable_source_documents": len(usable_documents),
        "complete_question_answer_pairs": sum(sources.values()) - sum(missing.values()),
        "unique_complete_pairs": len(unique_pairs),
        "unique_answer_texts": len(answer_hashes),
        "missing_question_or_answer_pairs": sum(missing.values()),
        "all_pairs_by_source": dict(sources),
        "missing_by_source": dict(missing),
    }


def prepare(archive: Path, output_dir: Path, seed: int = 5565) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = {name: [] for name in ("train", "validation", "test")}
    seen_pairs = set()
    missing = 0
    duplicate_pairs = 0
    for row in extract_records(archive):
        if not row["question"] or not row["answer"]:
            missing += 1
            continue
        key = (row["question"].casefold(), row["answer"].casefold())
        if key in seen_pairs:
            duplicate_pairs += 1
            continue
        seen_pairs.add(key)
        pairs[split_name(row["group"], seed)].append(row)

    cpt_texts = {}
    seen_answers = set()
    for split in ("train", "validation"):
        texts = []
        for row in pairs[split]:
            answer_key = sha256(row["answer"].casefold().encode()).hexdigest()
            if answer_key in seen_answers:
                continue
            seen_answers.add(answer_key)
            texts.append({"text": row["answer"], "source": row["source"], "document": row["document"]})
        cpt_texts[split] = texts

    for split, rows in pairs.items():
        _write_jsonl(output_dir / f"qa_{split}.jsonl", rows)
    for split, rows in cpt_texts.items():
        _write_jsonl(output_dir / f"cpt_{split}.jsonl", rows)

    manifest = {
        "source_archive": str(archive.resolve()),
        "source_sha256": sha256(archive.read_bytes()).hexdigest(),
        "seed": seed,
        "split_unit": "source URL (or source/document ID)",
        "removed_missing_pairs": missing,
        "removed_exact_duplicate_pairs": duplicate_pairs,
        "qa_counts": {name: len(rows) for name, rows in pairs.items()},
        "cpt_counts": {name: len(rows) for name, rows in cpt_texts.items()},
        "cpt_source": "unique complete answers in train/validation only; no test content",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
