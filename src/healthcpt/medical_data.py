"""Download and prepare the project's medical text corpus.

This module uses Python's standard library so the same commands work on a
server after ``uv sync --locked``. Raw downloads stay under ``data/raw`` and
prepared files are written to a new versioned directory.
"""

from collections import Counter
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import json
import random
import re
import time
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile
from xml.etree import ElementTree as ET


MEDLINEPLUS_XML_PAGE = "https://medlineplus.gov/xml.html"
PMC_BUCKET_URL = "https://pmc-oa-opendata.s3.amazonaws.com"
EUTILS_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
USER_AGENT = "DAT5565-HealthCPT/0.1 (academic course project)"
XML_NAMESPACE = "{http://www.w3.org/XML/1998/namespace}lang"
CPT_MEDQUAD_LIMIT = 6000
CPT_TEXT_CHUNK_WORDS = 200
SKIP_ARTICLE_TAGS = {
    "fig", "table-wrap", "table", "ref-list", "disp-formula",
    "inline-formula", "graphic", "media", "supplementary-material",
}


def clean_text(value: str) -> str:
    """Decode entities and turn XML whitespace into readable spaces."""
    return re.sub(r"\s+", " ", unescape(value or "")).strip()


class _HTMLVisibleText(HTMLParser):
    """Collect readable text from HTML embedded inside XML text fields."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def clean_html_text(value: str) -> str:
    """Remove embedded HTML tags while keeping their visible text."""
    parser = _HTMLVisibleText()
    parser.feed(value or "")
    parser.close()
    return clean_text(" ".join(parser.parts))


def normalized_text(value: str) -> str:
    """Normalize text for exact duplicate and held-out overlap checks."""
    return clean_text(value).casefold()


def _tag_name(tag: str) -> str:
    """Return an XML tag name without an optional namespace."""
    return tag.rsplit("}", 1)[-1]


def _element_text(element: ET.Element, skip_tags: set[str] | None = None) -> str:
    """Collect readable XML text, optionally skipping noisy subtrees."""
    skipped = skip_tags or set()
    pieces: list[str] = []

    def visit(node: ET.Element) -> None:
        if node.text:
            pieces.append(node.text)
        for child in node:
            if _tag_name(child.tag) not in skipped:
                visit(child)
            if child.tail:
                pieces.append(child.tail)

    visit(element)
    return clean_text(" ".join(pieces))


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSON-lines file into a list of records."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_bytes(url: str, timeout: int = 60) -> bytes:
    """Fetch a URL, retrying a few times if the connection drops."""
    for attempt in range(4):
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except URLError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not download {url}")


class _MedlinePlusZipLinks(HTMLParser):
    """Find dated compressed Health Topic XML links on the official page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if not href:
            return
        absolute_url = urljoin(MEDLINEPLUS_XML_PAGE, href)
        match = re.search(r"mplus_topics_compressed_(\d{4}-\d{2}-\d{2})\.zip$", urlsplit(absolute_url).path)
        if match:
            self.links.append((match.group(1), absolute_url))


def download_medlineplus(output_dir: Path) -> dict:
    """Download the newest official compressed MedlinePlus Health Topic XML."""
    output_dir.mkdir(parents=True, exist_ok=True)
    page = _download_bytes(MEDLINEPLUS_XML_PAGE).decode("utf-8", errors="replace")
    parser = _MedlinePlusZipLinks()
    parser.feed(page)
    if not parser.links:
        raise RuntimeError("Could not find a dated Health Topic ZIP on the MedlinePlus XML page")

    date, url = max(parser.links, key=lambda item: item[0])
    filename = Path(urlsplit(url).path).name
    destination = output_dir / filename
    if not destination.exists():
        destination.write_bytes(_download_bytes(url, timeout=120))

    return {
        "source": "MedlinePlus Health Topic XML",
        "date": date,
        "url": url,
        "file": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": _sha256_file(destination),
    }


def _allowed_pmc_license(license_code: str) -> bool:
    """Allow only simple CC BY and CC0 licenses for this project sample."""
    compact = re.sub(r"[^A-Z0-9]", "", (license_code or "").upper())
    if compact.startswith("CC0"):
        return True
    return compact.startswith("CCBY") and not any(flag in compact for flag in ("NC", "SA", "ND"))


def _pmc_search(topic: str, result_limit: int) -> list[str]:
    """Search PMC for English OA articles with CC BY or CC0 licenses."""
    query = (
        f'"{topic}"[Title/Abstract] AND open_access[Filter] '
        "AND (cc0_license[Filter] OR cc_by_license[Filter]) AND English[Language]"
    )
    parameters = urlencode({"db": "pmc", "term": query, "retmax": result_limit, "retmode": "json"})
    response = json.loads(_download_bytes(f"{EUTILS_ESEARCH_URL}?{parameters}").decode("utf-8"))
    return response.get("esearchresult", {}).get("idlist", [])


def _pmc_metadata_keys(pmc_numeric_id: str) -> list[str]:
    """List metadata JSON files for a PMC article's available versions."""
    parameters = urlencode({"list-type": "2", "prefix": f"metadata/PMC{pmc_numeric_id}."})
    xml = _download_bytes(f"{PMC_BUCKET_URL}/?{parameters}")
    root = ET.fromstring(xml)
    return [
        element.text for element in root.iter()
        if _tag_name(element.tag) == "Key" and element.text and element.text.endswith(".json")
    ]


def _select_pmc_metadata(pmc_numeric_id: str) -> dict | None:
    """Choose an accessible, non-retracted CC BY/CC0 article version."""
    candidates = []
    for key in _pmc_metadata_keys(pmc_numeric_id):
        url = f"{PMC_BUCKET_URL}/{key}"
        metadata = json.loads(_download_bytes(url).decode("utf-8"))
        if not metadata.get("is_pmc_openaccess") or metadata.get("is_retracted"):
            continue
        if not _allowed_pmc_license(metadata.get("license_code", "")):
            continue
        candidates.append(metadata)
    if not candidates:
        return None
    # Prefer the final published article when both it and a manuscript exist.
    candidates.sort(key=lambda item: (bool(item.get("is_manuscript")), int(item.get("version", 1))))
    return candidates[0]


def _https_article_url(url: str) -> str:
    """Turn the public S3 URL in PMC metadata into its HTTPS download URL."""
    prefix = "s3://pmc-oa-opendata/"
    if url.startswith(prefix):
        return f"{PMC_BUCKET_URL}/{url[len(prefix):]}"
    return url


def download_pmc_sample(
    qa_train_path: Path,
    output_dir: Path,
    topic_limit: int = 20,
    articles_per_topic: int = 30,
) -> dict:
    """Download a small licensed PMC sample based on MedQuAD training topics.

    This deliberately does not download all of PMC OA. The selected PMCID,
    version, topic, license, and file hash are written to a manifest.
    """
    if not qa_train_path.exists():
        raise FileNotFoundError(f"MedQuAD training file not found: {qa_train_path}")
    if topic_limit < 1 or articles_per_topic < 1:
        raise ValueError("topic_limit and articles_per_topic must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    cached_metadata = {}
    for metadata_path in output_dir.glob("PMC*.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            pmcid = metadata.get("pmcid", "")
            version = int(metadata.get("version", 1))
            if (
                pmcid
                and metadata_path.stem == f"{pmcid}.{version}"
                and metadata_path.with_suffix(".xml").exists()
                and metadata.get("is_pmc_openaccess")
                and not metadata.get("is_retracted")
                and _allowed_pmc_license(metadata.get("license_code", ""))
            ):
                cached_metadata[pmcid.removeprefix("PMC")] = metadata
        except (OSError, ValueError, json.JSONDecodeError):
            continue

    topics = Counter(
        clean_text(row.get("topic", ""))
        for row in _read_jsonl(qa_train_path)
        if clean_text(row.get("topic", ""))
    )
    selected_topics = [topic for topic, _ in topics.most_common(topic_limit)]
    selected_ids: set[str] = set()
    manifest_rows: list[dict] = []
    cached_articles_reused = 0
    per_topic_counts: dict[str, int] = {}

    for topic in selected_topics:
        # Request extra IDs in case some versions are retracted or have an
        # unexpected license code in the article metadata.
        candidate_ids = _pmc_search(topic, articles_per_topic * 5)
        downloaded_for_topic = 0
        for numeric_id in candidate_ids:
            if numeric_id in selected_ids:
                continue
            try:
                metadata = cached_metadata.get(numeric_id)
                if metadata:
                    cached_articles_reused += 1
                else:
                    metadata = _select_pmc_metadata(numeric_id)
            except Exception:
                # A single unavailable metadata record should not stop the sample.
                continue
            if not metadata:
                continue

            pmcid = metadata["pmcid"]
            version = int(metadata.get("version", 1))
            article_id = f"{pmcid}.{version}"
            xml_path = output_dir / f"{article_id}.xml"
            metadata_path = output_dir / f"{article_id}.json"
            if not xml_path.exists():
                xml_bytes = _download_bytes(_https_article_url(metadata["xml_url"]), timeout=120)
                xml_path.write_bytes(xml_bytes)
            metadata["healthcpt_search_topic"] = topic
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            manifest_rows.append({
                "pmcid": pmcid,
                "version": version,
                "title": metadata.get("title", ""),
                "license": metadata.get("license_code", ""),
                "search_topic": topic,
                "xml_file": xml_path.name,
                "xml_sha256": _sha256_file(xml_path),
            })
            selected_ids.add(numeric_id)
            downloaded_for_topic += 1
            if downloaded_for_topic >= articles_per_topic:
                break
        per_topic_counts[topic] = downloaded_for_topic
        # NCBI asks clients to keep E-utilities requests below three per second.
        time.sleep(0.4)

    manifest = {
        "source": "PMC Article Datasets AWS",
        "retrieval_query": "MedQuAD training topics AND English AND Open Access AND CC BY/CC0",
        "topic_limit": topic_limit,
        "articles_per_topic": articles_per_topic,
        "per_topic_counts": per_topic_counts,
        "article_count": len(manifest_rows),
        "cached_articles_reused": cached_articles_reused,
        "articles": manifest_rows,
    }
    (output_dir / "pmc_sample_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {key: value for key, value in manifest.items() if key != "articles"}


def download_medical_sources(
    qa_train_path: Path,
    output_dir: Path,
    topic_limit: int = 20,
    articles_per_topic: int = 30,
) -> dict:
    """Download MedlinePlus XML and a bounded PMC article sample."""
    medlineplus = download_medlineplus(output_dir / "medlineplus")
    pmc = download_pmc_sample(
        qa_train_path,
        output_dir / "pmc_oa",
        topic_limit=topic_limit,
        articles_per_topic=articles_per_topic,
    )
    return {"medlineplus": medlineplus, "pmc": pmc}


def _canonical_url(url: str) -> str:
    """Normalize known MedlinePlus URL redirects for overlap checks."""
    parsed = urlsplit((url or "").strip())
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path
    if host == "nlm.nih.gov" and path.startswith("/medlineplus/"):
        host = "medlineplus.gov"
        path = path[len("/medlineplus"):]
    return f"https://{host}{path}".rstrip("/").casefold() if host else ""


def _stable_split(document_id: str, seed: int = 5565) -> str:
    """Put about one in ten whole source documents in CPT validation."""
    value = int.from_bytes(sha256(f"{seed}:{document_id}".encode()).digest()[:8], "big") / 2**64
    return "validation" if value >= 0.9 else "train"


def _medlineplus_records(
    zip_path: Path,
    train_urls: set[str],
    heldout_urls: set[str],
) -> tuple[list[dict], dict]:
    """Extract English public-domain health-topic summaries from the ZIP."""
    records = []
    stats = Counter()
    with ZipFile(zip_path) as archive:
        xml_names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        topic_xml = None
        for name in xml_names:
            root = ET.fromstring(archive.read(name))
            if _tag_name(root.tag) == "health-topics":
                topic_xml = root
                break
        if topic_xml is None:
            raise ValueError(f"No Health Topic XML found in {zip_path}")

        for topic in topic_xml:
            if _tag_name(topic.tag) != "health-topic":
                continue
            stats["topics_seen"] += 1
            if topic.attrib.get("language", "").casefold() != "english":
                stats["non_english"] += 1
                continue
            summary = next((node for node in topic if _tag_name(node.tag) == "full-summary"), None)
            title = clean_text(topic.attrib.get("title", ""))
            raw_body = _element_text(summary) if summary is not None else ""
            body = clean_html_text(raw_body)
            if body != raw_body:
                stats["embedded_html_cleaned"] += 1
            if not title or not body:
                stats["missing_summary"] += 1
                continue
            url = topic.attrib.get("url", "")
            canonical = _canonical_url(url)
            if canonical and canonical in heldout_urls:
                stats["heldout_source_pages"] += 1
                continue
            split = "train" if canonical in train_urls else _stable_split(canonical or title)
            records.append({
                "text": f"{title}\n{body}",
                "source": "MedlinePlus",
                "document": topic.attrib.get("id", canonical or title),
                "url": url,
                "license": "Public domain health-topic summary; cite MedlinePlus/NLM",
                "split": split,
            })
            stats["usable_english_topics"] += 1
    return records, dict(stats)


def _first_element(root: ET.Element, name: str) -> ET.Element | None:
    return next((node for node in root.iter() if _tag_name(node.tag) == name), None)


def _pmc_records(pmc_dir: Path, heldout_answers: set[str]) -> tuple[list[dict], dict]:
    """Extract article title, abstract, and narrative body from saved JATS XML."""
    records = []
    stats = Counter()
    long_heldout = [answer for answer in heldout_answers if len(answer) >= 160]

    selection_path = pmc_dir / "pmc_sample_manifest.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"Run download-medical-sources first; missing {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    for selected in selection.get("articles", []):
        xml_path = pmc_dir / selected["xml_file"]
        stats["xml_files_seen"] += 1
        if not xml_path.exists():
            stats["missing_xml"] += 1
            continue
        if selected.get("xml_sha256") and _sha256_file(xml_path) != selected["xml_sha256"]:
            stats["xml_hash_mismatch"] += 1
            continue
        metadata_path = xml_path.with_suffix(".json")
        if not metadata_path.exists():
            stats["missing_metadata"] += 1
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        license_code = metadata.get("license_code", "")
        if not metadata.get("is_pmc_openaccess") or metadata.get("is_retracted") or not _allowed_pmc_license(license_code):
            stats["license_or_retraction_filtered"] += 1
            continue
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            stats["bad_xml"] += 1
            continue

        article_type = root.attrib.get("article-type", "").casefold()
        if any(flag in article_type for flag in ("retraction", "correction", "expression-of-concern")):
            stats["article_type_filtered"] += 1
            continue
        language = root.attrib.get(XML_NAMESPACE, "").casefold()
        if language and not language.startswith("en"):
            stats["non_english"] += 1
            continue

        title_node = _first_element(root, "article-title")
        abstract_node = _first_element(root, "abstract")
        body_node = _first_element(root, "body")
        title = _element_text(title_node) if title_node is not None else ""
        abstract = _element_text(abstract_node) if abstract_node is not None else ""
        body = _element_text(body_node, SKIP_ARTICLE_TAGS) if body_node is not None else ""
        text = clean_text("\n".join(part for part in (title, abstract, body) if part))
        if not text:
            stats["empty_text"] += 1
            continue

        normalized = normalized_text(text)
        if any(answer in normalized for answer in long_heldout):
            stats["heldout_answer_overlap"] += 1
            continue
        pmcid = metadata.get("pmcid", xml_path.stem.rsplit(".", 1)[0])
        version = metadata.get("version", "1")
        document = f"{pmcid}.{version}"
        records.append({
            "text": text,
            "source": "PMC",
            "document": document,
            "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
            "license": license_code,
            "doi": metadata.get("doi", ""),
            "search_topic": metadata.get("healthcpt_search_topic", ""),
            "split": _stable_split(document),
        })
        stats["usable_articles"] += 1
    return records, dict(stats)


def _deduplicate_records(train_rows: list[dict], validation_rows: list[dict]) -> tuple[list[dict], list[dict], int]:
    """Remove repeated CPT text, giving the training split priority."""
    seen: set[str] = set()
    clean_train = []
    removed = 0
    for row in train_rows:
        key = sha256(normalized_text(row["text"]).encode()).hexdigest()
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        clean_train.append(row)

    clean_validation = []
    for row in validation_rows:
        key = sha256(normalized_text(row["text"]).encode()).hexdigest()
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        clean_validation.append(row)
    return clean_train, clean_validation, removed


def _sample_records(rows: list[dict], limit: int, seed: int = 5565) -> list[dict]:
    """Choose a repeatable random subset while keeping the original order."""
    if limit < 1:
        raise ValueError("medquad_cpt_limit must be positive")
    if limit >= len(rows):
        return rows
    selected = set(random.Random(seed).sample(range(len(rows)), limit))
    return [row for index, row in enumerate(rows) if index in selected]


def _chunk_records(rows: list[dict], max_words: int) -> list[dict]:
    """Split long CPT texts into shorter word-bounded training examples."""
    if max_words < 1:
        raise ValueError("chunk_words must be positive")
    chunks = []
    for row in rows:
        words = row["text"].split()
        for chunk_index, start in enumerate(range(0, len(words), max_words)):
            chunk = dict(row)
            chunk["text"] = " ".join(words[start:start + max_words])
            chunk["chunk_index"] = chunk_index
            chunks.append(chunk)
    return chunks


def _summarize_cpt_rows(rows: list[dict]) -> dict:
    """Report simple counts without implying whitespace words are model tokens."""
    return {
        "records_by_source": dict(Counter(row.get("source", "unknown") for row in rows)),
        "documents_by_source": {
            source: len({row.get("document", "") for row in rows if row.get("source") == source})
            for source in sorted({row.get("source", "unknown") for row in rows})
        },
        "licenses": dict(Counter(row.get("license", "not recorded") for row in rows)),
        "characters_by_source": {
            source: sum(len(row["text"]) for row in rows if row.get("source") == source)
            for source in sorted({row.get("source", "unknown") for row in rows})
        },
        "whitespace_words_by_source": {
            source: sum(len(row["text"].split()) for row in rows if row.get("source") == source)
            for source in sorted({row.get("source", "unknown") for row in rows})
        },
    }


def _clean_evaluation_rows(
    rows: list[dict],
    train_questions: set[str],
    train_answers: set[str],
    cpt_texts: list[str],
    external_cpt_texts: list[str],
    cpt_urls: set[str],
) -> tuple[list[dict], dict]:
    """Remove held-out QA whose source, question, or answer was seen in CPT/SFT train."""
    clean = []
    removed = Counter()
    long_cpt_texts = [normalized_text(text) for text in external_cpt_texts if len(text) >= 1000]
    cpt_text_hashes = {sha256(normalized_text(text).encode()).hexdigest() for text in cpt_texts}

    for row in rows:
        question = normalized_text(row.get("question", ""))
        answer = normalized_text(row.get("answer", ""))
        url = _canonical_url(row.get("url", ""))
        answer_hash = sha256(answer.encode()).hexdigest()
        if url and url in cpt_urls:
            removed["source_url_in_cpt_train"] += 1
        elif question and question in train_questions:
            removed["question_seen_in_sft_train"] += 1
        elif answer and (answer in train_answers or answer_hash in cpt_text_hashes):
            removed["answer_seen_in_training_text"] += 1
        elif len(answer) >= 160 and any(answer in text for text in long_cpt_texts):
            removed["answer_found_inside_cpt_document"] += 1
        else:
            clean.append(row)
    return clean, dict(removed)


def prepare_medical_corpus(
    medquad_dir: Path,
    source_dir: Path,
    output_dir: Path,
    medquad_cpt_limit: int = CPT_MEDQUAD_LIMIT,
    chunk_words: int = CPT_TEXT_CHUNK_WORDS,
) -> dict:
    """Balance CPT sources, split long texts, and create clean eval files."""
    qa_train = _read_jsonl(medquad_dir / "qa_train.jsonl")
    qa_validation = _read_jsonl(medquad_dir / "qa_validation.jsonl")
    qa_test = _read_jsonl(medquad_dir / "qa_test.jsonl")
    medquad_cpt_train = _read_jsonl(medquad_dir / "cpt_train.jsonl")
    train_questions = {normalized_text(row.get("question", "")) for row in qa_train}
    train_answers = {normalized_text(row.get("answer", "")) for row in qa_train}

    validation_urls = {_canonical_url(row.get("url", "")) for row in qa_validation if row.get("url")}
    test_urls = {_canonical_url(row.get("url", "")) for row in qa_test if row.get("url")}
    train_urls = {_canonical_url(row.get("url", "")) for row in qa_train if row.get("url")}
    heldout_urls = validation_urls | test_urls
    heldout_answers = {
        normalized_text(row.get("answer", ""))
        for row in qa_validation + qa_test if row.get("answer")
    }

    medlineplus_dir = source_dir / "medlineplus"
    zip_files = sorted(medlineplus_dir.glob("mplus_topics_compressed_*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"Download the MedlinePlus XML first into {medlineplus_dir}")
    medlineplus_rows, medlineplus_stats = _medlineplus_records(zip_files[-1], train_urls, heldout_urls)
    pmc_dir = source_dir / "pmc_oa"
    pmc_rows, pmc_stats = _pmc_records(pmc_dir, heldout_answers)

    # These unique MedQuAD answers come only from the SFT training split.
    medquad_cpt_sample = _sample_records(medquad_cpt_train, medquad_cpt_limit)
    train_candidates = [
        dict(
            row,
            dataset="MedQuAD",
            license="CC BY 4.0 (MedQuAD dataset; retain source attribution)",
        )
        for row in medquad_cpt_sample
    ]
    validation_candidates: list[dict] = []
    for row in medlineplus_rows + pmc_rows:
        split = row.pop("split")
        (validation_candidates if split == "validation" else train_candidates).append(row)

    cpt_train_documents, cpt_validation_documents, duplicate_texts_removed = _deduplicate_records(
        train_candidates, validation_candidates
    )
    all_cpt_texts = [row["text"] for row in cpt_train_documents + cpt_validation_documents]
    external_cpt_texts = [
        row["text"]
        for row in cpt_train_documents + cpt_validation_documents
        if row.get("dataset") != "MedQuAD"
    ]
    cpt_urls = {
        _canonical_url(row.get("url", ""))
        for row in cpt_train_documents + cpt_validation_documents
        if row.get("url")
    }
    qa_validation_eval, validation_removed = _clean_evaluation_rows(
        qa_validation, train_questions, train_answers, all_cpt_texts, external_cpt_texts, cpt_urls
    )
    qa_test_eval, test_removed = _clean_evaluation_rows(
        qa_test, train_questions, train_answers, all_cpt_texts, external_cpt_texts, cpt_urls
    )

    # KerasHub limits each string to the configured sequence length. Split long
    # source texts here so their later sections are not silently dropped.
    cpt_train = _chunk_records(cpt_train_documents, chunk_words)
    cpt_validation = _chunk_records(cpt_validation_documents, chunk_words)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "cpt_train.jsonl", cpt_train)
    _write_jsonl(output_dir / "cpt_validation.jsonl", cpt_validation)
    _write_jsonl(output_dir / "qa_validation_eval.jsonl", qa_validation_eval)
    _write_jsonl(output_dir / "qa_test_eval.jsonl", qa_test_eval)

    pmc_manifest_path = pmc_dir / "pmc_sample_manifest.json"
    manifest = {
        "seed": 5565,
        "medquad_manifest_sha256": _sha256_file(medquad_dir / "manifest.json"),
        "medlineplus_archive": zip_files[-1].name,
        "medlineplus_sha256": _sha256_file(zip_files[-1]),
        "pmc_sample_manifest": pmc_manifest_path.as_posix(),
        "pmc_sample_manifest_sha256": _sha256_file(pmc_manifest_path),
        "pmc_article_count": len(json.loads(pmc_manifest_path.read_text(encoding="utf-8")).get("articles", [])),
        "cpt_text_preparation": {
            "medquad_answer_texts_available": len(medquad_cpt_train),
            "medquad_answer_texts_selected": len(medquad_cpt_sample),
            "medquad_sample_seed": 5565,
            "target_words_per_chunk": chunk_words,
        },
        "cpt_train": _summarize_cpt_rows(cpt_train),
        "cpt_validation": _summarize_cpt_rows(cpt_validation),
        "medlineplus_cleaning": medlineplus_stats,
        "pmc_cleaning": pmc_stats,
        "duplicate_cpt_texts_removed": duplicate_texts_removed,
        "qa_validation_rows": {"before": len(qa_validation), "after": len(qa_validation_eval), "removed": validation_removed},
        "qa_test_rows": {"before": len(qa_test), "after": len(qa_test_eval), "removed": test_removed},
        "outputs": [
            "cpt_train.jsonl", "cpt_validation.jsonl",
            "qa_validation_eval.jsonl", "qa_test_eval.jsonl",
        ],
        "notes": [
            "Original MedQuAD QA splits were not modified.",
            "CPT validation is built from external MedlinePlus/PMC text, not MedQuAD QA validation answers.",
            "MedlinePlus pages matching MedQuAD validation/test URLs are excluded from CPT.",
            "Evaluation rows with repeated questions/answers or answer text found in CPT documents are excluded from clean evaluation files.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
