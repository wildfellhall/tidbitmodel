"""Download, clean, split by work family, and tokenize corpus.py into passages.

The parser selects prose and Q&A, removes exact duplicate paragraphs and
Gutenberg wrappers, and reflows lines. This formatting heuristic does not
fact-check books or remove every index, table, or near-duplicate edition.
"""

import ast
import bisect
import hashlib
import io
import json
import random
import re
import shutil
import tempfile
import time
import tokenize
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|endoftext|>"]
START = re.compile(
    r"^\*\*\*\s*START OF .*?PROJECT GUTENBERG.*?\*\*\*[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
END = re.compile(
    r"^\*\*\*\s*END OF .*?PROJECT GUTENBERG.*?\*\*\*[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_sources(path):
    """Read a literal sources list without executing any Python from the file."""
    path = Path(path)
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "sources"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise ValueError(
            "The corpus file must contain exactly one literal sources = [...] assignment."
        )
    values = ast.literal_eval(assignments[0].value)
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(v, str) and v for v in values)
    ):
        raise ValueError(
            "sources must be a nonempty list of URL or local-file strings."
        )
    comments = {
        token.start[0]: token.string.lstrip("# ").strip()
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    }
    entries, seen = [], set()
    for value, node in zip(values, assignments[0].value.elts):
        if value in seen:
            continue
        seen.add(value)
        parsed = urlparse(value)
        if parsed.scheme not in {"", "http", "https"}:
            raise ValueError(f"Unsupported source: {value}")
        if not parsed.scheme:
            value = str((path.parent / value).resolve())
        entries.append(
            {
                "source": value,
                "title": comments.get(node.lineno, Path(parsed.path).name),
            }
        )
    return entries


def download_url(source, mirror):
    parsed = urlparse(source)
    if parsed.hostname in {"gutenberg.org", "www.gutenberg.org"}:
        match = re.fullmatch(r"/cache/epub/(\d+)/pg(\d+)\.txt", parsed.path)
        if match:
            if match[1] != match[2]:
                raise ValueError(f"Mismatched Gutenberg IDs in {source}")
            return f"{mirror.rstrip('/')}/{match[1]}/pg{match[1]}.txt"
    return source


def clean_text(raw, require_wrappers=False):
    raw = unicodedata.normalize(
        "NFC", raw.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    )
    if raw.lstrip().lower().startswith(("<!doctype html", "<html")):
        raise ValueError("Expected plain text, received HTML.")
    start, end = START.search(raw), END.search(raw)
    if require_wrappers and (not start or not end):
        raise ValueError(
            "Gutenberg start/end markers are missing; refusing to train on a possible error page."
        )
    if start and end and start.end() >= end.start():
        raise ValueError("Gutenberg markers are out of order.")
    body = raw[start.end() if start else 0 : end.start() if end else len(raw)]
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", body):
        text = re.sub(r"\s+", " ", paragraph).strip()
        if text and not re.fullmatch(
            r"\[Illustration[^\]]*\]", text, flags=re.IGNORECASE
        ):
            paragraphs.append(text)
    cleaned = "\n\n".join(paragraphs)
    if len(cleaned) < 200:
        raise ValueError("Fewer than 200 characters remain after cleaning.")
    return cleaned + "\n"


def prose_paragraphs(text, min_words=8):
    """Conservative prose selection, including adjacent question/answer pairs.

    This is a formatting filter, not a factual or semantic quality classifier.
    All removed text remains available in the raw download cache.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    pending = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if re.match(r"^(?:Q\.|Q:|Question\b)", paragraph, re.IGNORECASE):
            pending.append(paragraph)
            continue
        if pending:
            paragraph = "\n".join(pending + [paragraph])
            pending = []
        if len(paragraph.split()) < min_words:
            continue
        if re.match(
            r"^(?:\[?transcriber|produced by|etext prepared|credits:)",
            paragraph,
            re.IGNORECASE,
        ):
            continue
        visible = sum(not ch.isspace() for ch in paragraph)
        letters = sum(ch.isalpha() for ch in paragraph)
        if not visible or letters / visible < 0.65:
            continue
        if not re.search(r"[.!?](?:\s|$|[\"”’\]])", paragraph):
            continue
        yield paragraph


def document_identity(raw, fallback):
    """Keep volumes/editions of the same titled work in one held-out split."""
    header = raw[: raw.find("*** START") if "*** START" in raw else 2000]
    title = re.search(r"^Title:\s*(.+)", header, re.MULTILINE)
    author = re.search(r"^Author:\s*(.+)", header, re.MULTILINE)
    title = title[1].strip() if title else fallback
    family = re.sub(
        r"\b(?:volumes?|vol\.?|parts?|books?)\s+[\divxlcdm]+\b.*",
        "",
        title,
        flags=re.IGNORECASE,
    )
    family = re.sub(r"\W+", " ", family.casefold()).strip()
    return title, family + "|" + (author[1].casefold().strip() if author else "")


def passage_tokens(text, backend, context, bos, eos):
    """Prefer sentence boundaries; never combine paragraphs from different books.

    A long sentence can require a hard cut. Such cuts do not receive an EOS
    target: the model must not learn that arbitrary token limits end a thought.
    """
    encoding = backend.encode(text, add_special_tokens=False)
    ids = encoding.ids
    ends = [end for _, end in encoding.offsets]
    boundaries = {
        bisect.bisect_right(ends, m.end())
        for m in re.finditer(r"[.!?][\"”’\])]*(?=\s|$)", text)
    }
    boundaries.add(len(ids))
    start, limit = 0, context - 2
    while start < len(ids):
        cap = min(start + limit, len(ids))
        candidates = [b for b in boundaries if start + max(1, limit // 4) <= b <= cap]
        end = max(candidates) if candidates else cap
        yield [bos] + ids[start:end] + ([eos] if end in boundaries else [])
        start = end


def fetch_source(session, source, mirror, destination, delay):
    """Try the generated mirror, then its legacy text paths on a missing file."""
    primary = download_url(source, mirror)
    candidates = [primary]
    match = re.search(r"/cache/epub/(\d+)/pg\d+\.txt$", urlparse(source).path)
    if match and mirror.rstrip("/").endswith("/gutenberg-epub"):
        book = match[1]
        legacy = mirror.rstrip("/").rsplit("/", 1)[0] + "/gutenberg/"
        legacy += "/".join(book[:-1]) + "/" + book + "/" + book
        candidates.extend([legacy + "-0.txt", legacy + ".txt"])
    part = destination.with_suffix(".part")
    try:
        for index, url in enumerate(candidates):
            try:
                with session.get(url, timeout=(10, 60), stream=True) as response:
                    if response.status_code == 404 and index + 1 < len(candidates):
                        continue
                    response.raise_for_status()
                    with part.open("wb") as stream:
                        for chunk in response.iter_content(chunk_size=65536):
                            stream.write(chunk)
                clean_text(
                    part.read_text(encoding="utf-8-sig"), require_wrappers=bool(match)
                )
                part.replace(destination)
                if index:
                    print(f"Using legacy Gutenberg text: {url}", flush=True)
                return
            finally:
                time.sleep(delay)
    finally:
        part.unlink(missing_ok=True)


def split_documents(documents, validation_fraction, test_fraction, seed):
    if (
        validation_fraction <= 0
        or test_fraction <= 0
        or validation_fraction + test_fraction >= 1
    ):
        raise ValueError(
            "Validation/test fractions must be positive and sum to less than one."
        )
    if len(documents) < 3:
        raise ValueError(
            "At least three distinct books are needed for train/validation/test splits."
        )
    groups = {}
    for document in documents:
        groups.setdefault(document.get("group", document["sha256"]), []).append(
            document
        )
    grouped = list(groups.values())
    if len(grouped) < 3:
        raise ValueError(
            "At least three independent work families are needed for held-out splits."
        )
    random.Random(seed).shuffle(grouped)
    result = []
    for split, fraction in (
        ("validation", validation_fraction),
        ("test", test_fraction),
    ):
        target, count = max(1, round(len(documents) * fraction)), 0
        reserved = 2 if split == "validation" else 1
        while count < target and len(grouped) > reserved:
            group = grouped.pop(0)
            for document in group:
                document["split"] = split
            result.extend(group)
            count += len(group)
    for group in grouped:
        for document in group:
            document["split"] = "train"
        result.extend(group)
    return result


def text_chunks(path, size=16384):
    """Bound tokenizer memory even for multi-volume reference books."""
    with Path(path).open(encoding="utf-8") as stream:
        while chunk := stream.read(size):
            yield chunk


def prepare(args):
    from tokenizers import (
        Tokenizer,
        decoders,
        models,
        pre_tokenizers,
        processors,
        trainers,
    )
    from transformers import PreTrainedTokenizerFast

    if args.context < 16 or args.min_document_words < 0 or args.min_paragraph_words < 1:
        raise ValueError(
            "Context must be >=16 and text-filter thresholds must be nonnegative."
        )
    if not 259 <= args.vocab_size <= 65535:
        raise ValueError(
            "vocab-size must be between 259 and 65535 for byte BPE and uint16 storage."
        )
    if args.download_delay < 0 or args.max_sources < 0:
        raise ValueError("Download delay and max-sources must be nonnegative.")
    if (
        args.validation_fraction <= 0
        or args.test_fraction <= 0
        or args.validation_fraction + args.test_fraction >= 1
    ):
        raise ValueError(
            "Validation/test fractions must be positive and sum to less than one."
        )
    output, cache = Path(args.data_dir), Path(args.cache_dir)
    if output.exists():
        raise ValueError(
            f"{output} already exists. Use a new --data-dir to preserve existing splits."
        )
    entries = load_sources(args.corpus)
    if args.max_sources:
        entries = entries[: args.max_sources]
    output.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "tidbitmodel/0.1 (corpus preparation)"
    try:
        with tempfile.TemporaryDirectory(
            prefix=".prepare-", dir=output.parent
        ) as temporary:
            staging = Path(temporary)
            (staging / "clean").mkdir()
            documents, failures, duplicates, hashes = [], [], [], set()
            paragraph_hashes, exclusions = set(), []
            filter_counts = {
                "input_words": 0,
                "retained_words": 0,
                "duplicate_paragraphs": 0,
            }
            for index, entry in enumerate(entries, 1):
                source = entry["source"]
                try:
                    is_remote = urlparse(source).scheme in {"http", "https"}
                    is_gutenberg = urlparse(source).hostname in {
                        "gutenberg.org",
                        "www.gutenberg.org",
                    }
                    if is_remote:
                        raw_path = cache / (
                            hashlib.sha256(source.encode()).hexdigest() + ".txt"
                        )
                        if not raw_path.exists():
                            fetch_source(
                                session,
                                source,
                                args.mirror,
                                raw_path,
                                args.download_delay,
                            )
                    else:
                        raw_path = Path(source)
                    raw = raw_path.read_text(encoding="utf-8-sig")
                    text = clean_text(raw, require_wrappers=is_remote and is_gutenberg)
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    if digest in hashes:
                        duplicates.append(source)
                        continue
                    hashes.add(digest)
                    filter_counts["input_words"] += len(text.split())
                    kept, pending_hashes = [], set()
                    for paragraph in prose_paragraphs(text, args.min_paragraph_words):
                        phash = hashlib.sha256(
                            re.sub(r"\s+", " ", paragraph).casefold().encode()
                        ).hexdigest()
                        if phash in paragraph_hashes or phash in pending_hashes:
                            filter_counts["duplicate_paragraphs"] += 1
                            continue
                        kept.append(paragraph)
                        pending_hashes.add(phash)
                    retained_words = sum(len(p.split()) for p in kept)
                    if not kept or retained_words < args.min_document_words:
                        exclusions.append(
                            {
                                "source": source,
                                "title": entry["title"],
                                "retained_words": retained_words,
                                "reason": "Too little prose after filtering and deduplication",
                            }
                        )
                        print(
                            f"[{index}/{len(entries)}] Excluded {entry['title']}: only {retained_words:,} retained prose words",
                            flush=True,
                        )
                        continue
                    paragraph_hashes.update(pending_hashes)
                    filter_counts["retained_words"] += retained_words
                    text = "\n\n".join(kept) + "\n"
                    title, group = document_identity(raw, entry["title"])
                    relative = f"clean/{digest}.txt"
                    (staging / relative).write_text(text, encoding="utf-8")
                    documents.append(
                        {
                            **entry,
                            "title": title,
                            "group": group,
                            "raw_sha256": digest,
                            "sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "path": relative,
                            "characters": len(text),
                        }
                    )
                    print(
                        f"[{index}/{len(entries)}] Parsed {entry['title']} ({len(text):,} characters)",
                        flush=True,
                    )
                except (
                    OSError,
                    UnicodeError,
                    ValueError,
                    requests.RequestException,
                ) as exc:
                    failures.append({"source": source, "error": str(exc)})
                    print(
                        f"[{index}/{len(entries)}] FAILED {source}: {exc}", flush=True
                    )
            write_json(
                cache / "last-prepare-report.json",
                {
                    "failures": failures,
                    "duplicate_sources": duplicates,
                    "excluded_documents": exclusions,
                    "filter_counts": filter_counts,
                },
            )
            if failures and not args.skip_errors:
                raise ValueError(
                    f"{len(failures)} sources failed; see {cache / 'last-prepare-report.json'}. Fix them or explicitly use --skip-errors."
                )
            documents = split_documents(
                documents, args.validation_fraction, args.test_fraction, args.seed
            )
            backend = Tokenizer(models.BPE())
            backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            backend.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=args.vocab_size,
                min_frequency=2,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                special_tokens=SPECIAL_TOKENS,
            )
            training_text = (
                chunk
                for document in documents
                if document["split"] == "train"
                for chunk in text_chunks(staging / document["path"])
            )
            print(
                "Training a byte-level BPE tokenizer on training books only...",
                flush=True,
            )
            backend.train_from_iterator(training_text, trainer=trainer)
            backend.post_processor = processors.TemplateProcessing(
                single="<|bos|> $A",
                pair="<|bos|> $A $B",
                special_tokens=[("<|bos|>", backend.token_to_id("<|bos|>"))],
            )
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend,
                pad_token=SPECIAL_TOKENS[0],
                bos_token=SPECIAL_TOKENS[1],
                eos_token=SPECIAL_TOKENS[2],
            )
            tokenizer.save_pretrained(staging / "tokenizer")
            counts, passage_counts, target_counts = {}, {}, {}
            for split in ("train", "validation", "test"):
                count = 0
                offsets = [0]
                with (staging / f"{split}.bin").open("wb") as stream:
                    for document in documents:
                        if document["split"] != split:
                            continue
                        before = count
                        document["first_passage"] = len(offsets) - 1
                        for paragraph in (
                            (staging / document["path"])
                            .read_text(encoding="utf-8")
                            .split("\n\n")
                        ):
                            for ids in passage_tokens(
                                paragraph,
                                backend,
                                args.context,
                                tokenizer.bos_token_id,
                                tokenizer.eos_token_id,
                            ):
                                np.asarray(ids, dtype="<u2").tofile(stream)
                                count += len(ids)
                                offsets.append(count)
                        document["tokens"] = count - before
                        document["passages"] = (
                            len(offsets) - 1 - document["first_passage"]
                        )
                counts[split] = count
                passage_counts[split] = len(offsets) - 1
                target_counts[split] = count - passage_counts[split]
                np.save(
                    staging / f"{split}.offsets.npy", np.asarray(offsets, dtype="<i8")
                )
            fingerprint = hashlib.sha256(
                (json.dumps(documents, sort_keys=True) + backend.to_str()).encode()
            ).hexdigest()
            metadata = {
                "format_version": 2,
                "context": args.context,
                "dtype": "<u2",
                "seed": args.seed,
                "vocab_size": len(tokenizer),
                "token_counts": counts,
                "passage_counts": passage_counts,
                "target_counts": target_counts,
                "filter_counts": filter_counts,
                "excluded_documents": exclusions,
                "fingerprint": fingerprint,
                "documents": documents,
                "failed_sources": failures,
                "duplicate_sources": duplicates,
                "split_unit": "work family; exact duplicate books and paragraphs removed before splitting",
            }
            write_json(staging / "metadata.json", metadata)
            # Publish only a complete dataset. Downloads remain reusable after failures.
            shutil.move(str(staging), str(output))
    finally:
        session.close()
    print(
        json.dumps(
            {
                "data_dir": str(output),
                "books": len(documents),
                "vocab_size": len(tokenizer),
                "tokens": counts,
                "passages": passage_counts,
                "target_tokens": target_counts,
                "excluded_documents": len(exclusions),
                "filter_counts": filter_counts,
            },
            indent=2,
        )
    )


class TokenBlocks:
    """Memory-mapped causal-LM sequences with each target counted once.

    Format 2 stores independent passages with BOS and sentence-aware EOS.
    Padding labels are masked. Legacy flat streams remain readable for tests.
    """

    def __init__(self, data_dir, split, context, pad_token_id):
        import torch

        self.torch = torch
        self.context, self.pad = context, pad_token_id
        metadata = json.loads((Path(data_dir) / "metadata.json").read_text())
        path = Path(data_dir) / f"{split}.bin"
        dtype = np.dtype(metadata["dtype"])
        if path.stat().st_size != metadata["token_counts"][split] * dtype.itemsize:
            raise ValueError(f"{path} does not match its recorded token count.")
        self.tokens = np.memmap(path, dtype=dtype, mode="r")
        self.offsets = None
        if metadata.get("format_version", 1) >= 2:
            if context != metadata["context"]:
                raise ValueError(
                    "Training context must match the prepared passage context; prepare a new dataset."
                )
            self.offsets = np.load(
                Path(data_dir) / f"{split}.offsets.npy", mmap_mode="r"
            )
            lengths = np.diff(self.offsets)
            if (
                self.offsets[0] != 0
                or self.offsets[-1] != len(self.tokens)
                or len(lengths) != metadata["passage_counts"][split]
                or np.any(lengths < 2)
                or np.any(lengths > context)
            ):
                raise ValueError("Invalid passage offsets or sequence lengths.")
        if context < 2 or len(self.tokens) < 2:
            raise ValueError(
                "Context and token stream must each contain at least two tokens."
            )

    def __len__(self):
        if self.offsets is not None:
            return len(self.offsets) - 1
        return (len(self.tokens) - 2) // (self.context - 1) + 1

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.offsets is None:
            start = index * (self.context - 1)
            end = start + self.context
        else:
            start, end = self.offsets[index : index + 2]
        chunk = np.asarray(self.tokens[start:end], dtype=np.int64)
        size = len(chunk)
        inputs = self.torch.full((self.context,), self.pad, dtype=self.torch.long)
        inputs[:size] = self.torch.from_numpy(chunk.copy())
        labels = inputs.clone()
        labels[size:] = -100
        mask = self.torch.zeros(self.context, dtype=self.torch.long)
        mask[:size] = 1
        return {"input_ids": inputs, "labels": labels, "attention_mask": mask}
