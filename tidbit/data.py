"""Download, clean, split by book, and tokenize the literal list in corpus.py.

The parser preserves historical prose, numbers, equations, and Q&A wording.
It removes Gutenberg wrappers and reflows lines; it does not fact-check the
books or claim to remove every index, table, or near-duplicate edition.
"""

import ast
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
    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    n_validation = max(1, round(len(shuffled) * validation_fraction))
    n_test = max(1, round(len(shuffled) * test_fraction))
    if n_validation + n_test >= len(shuffled):
        raise ValueError("Split fractions leave no training books.")
    for index, document in enumerate(shuffled):
        document["split"] = (
            "validation"
            if index < n_validation
            else "test"
            if index < n_validation + n_test
            else "train"
        )
    return shuffled


def text_chunks(path, size=16384):
    """Bound tokenizer memory even for multi-volume reference books."""
    with Path(path).open(encoding="utf-8") as stream:
        while chunk := stream.read(size):
            yield chunk


def prepare(args):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

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
            for index, entry in enumerate(entries, 1):
                source = entry["source"]
                try:
                    is_remote = urlparse(source).scheme in {"http", "https"}
                    is_gutenberg = urlparse(source).hostname in {
                        "gutenberg.org",
                        "www.gutenberg.org",
                    }
                    if is_remote:
                        url = download_url(source, args.mirror)
                        raw_path = cache / (
                            hashlib.sha256(source.encode()).hexdigest() + ".txt"
                        )
                        if not raw_path.exists():
                            part = raw_path.with_suffix(".part")
                            try:
                                with session.get(
                                    url, timeout=(10, 60), stream=True
                                ) as response:
                                    response.raise_for_status()
                                    with part.open("wb") as stream:
                                        for chunk in response.iter_content(
                                            chunk_size=65536
                                        ):
                                            stream.write(chunk)
                                # Validate downloaded content before making it a reusable cache entry.
                                clean_text(
                                    part.read_text(encoding="utf-8-sig"),
                                    require_wrappers=is_gutenberg,
                                )
                                part.replace(raw_path)
                            finally:
                                part.unlink(missing_ok=True)
                                time.sleep(args.download_delay)
                    else:
                        raw_path = Path(source)
                    raw = raw_path.read_text(encoding="utf-8-sig")
                    text = clean_text(raw, require_wrappers=is_remote and is_gutenberg)
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    if digest in hashes:
                        duplicates.append(source)
                        continue
                    hashes.add(digest)
                    relative = f"clean/{digest}.txt"
                    (staging / relative).write_text(text, encoding="utf-8")
                    documents.append(
                        {
                            **entry,
                            "sha256": digest,
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
                {"failures": failures, "duplicate_sources": duplicates},
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
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend,
                pad_token=SPECIAL_TOKENS[0],
                bos_token=SPECIAL_TOKENS[1],
                eos_token=SPECIAL_TOKENS[2],
            )
            tokenizer.save_pretrained(staging / "tokenizer")
            counts = {}
            for split in ("train", "validation", "test"):
                count = 0
                with (staging / f"{split}.bin").open("wb") as stream:
                    for document in documents:
                        if document["split"] != split:
                            continue
                        before = count
                        np.asarray([tokenizer.bos_token_id], dtype="<u2").tofile(stream)
                        count += 1
                        for chunk in text_chunks(staging / document["path"]):
                            ids = backend.encode(chunk, add_special_tokens=False).ids
                            np.asarray(ids, dtype="<u2").tofile(stream)
                            count += len(ids)
                        np.asarray([tokenizer.eos_token_id], dtype="<u2").tofile(stream)
                        count += 1
                        document["tokens"] = count - before
                counts[split] = count
            fingerprint = hashlib.sha256(
                (json.dumps(documents, sort_keys=True) + backend.to_str()).encode()
            ).hexdigest()
            metadata = {
                "format_version": 1,
                "dtype": "<u2",
                "seed": args.seed,
                "vocab_size": len(tokenizer),
                "token_counts": counts,
                "fingerprint": fingerprint,
                "documents": documents,
                "failed_sources": failures,
                "duplicate_sources": duplicates,
                "split_unit": "whole book; exact duplicate books removed before splitting",
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
            },
            indent=2,
        )
    )


class TokenBlocks:
    """Memory-mapped causal-LM sequences with each target counted once.

    Adjacent blocks overlap by one context token. The final short block is
    padded and its padding labels are masked. EOS/BOS delimit packed books.
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
        if context < 2 or len(self.tokens) < 2:
            raise ValueError(
                "Context and token stream must each contain at least two tokens."
            )

    def __len__(self):
        return (len(self.tokens) - 2) // (self.context - 1) + 1

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * (self.context - 1)
        chunk = np.asarray(self.tokens[start : start + self.context], dtype=np.int64)
        size = len(chunk)
        inputs = self.torch.full((self.context,), self.pad, dtype=self.torch.long)
        inputs[:size] = self.torch.from_numpy(chunk.copy())
        labels = inputs.clone()
        labels[size:] = -100
        mask = self.torch.zeros(self.context, dtype=self.torch.long)
        mask[:size] = 1
        return {"input_ids": inputs, "labels": labels, "attention_mask": mask}
