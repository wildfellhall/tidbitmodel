"""Offline regression and end-to-end tests: python -m unittest discover -s tests -v."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from main import main, parser
from tidbit.data import (
    TokenBlocks,
    clean_text,
    download_url,
    load_sources,
    prepare,
    split_documents,
    write_json,
)
from tidbit.training import build_model, sample_responses, train


class ParsingTests(unittest.TestCase):
    def test_literal_sources_are_not_executed_and_urls_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sources.py"
            path.write_text(
                'sources = ["book.txt", "book.txt"] # A book\nraise RuntimeError("must not execute")\n'
            )
            result = load_sources(path)
            self.assertEqual(len(result), 1)
            self.assertEqual(
                result[0]["source"], str((Path(folder) / "book.txt").resolve())
            )
            path.write_text('sources = __import__("os").listdir()')
            with self.assertRaises(ValueError):
                load_sources(path)

    def test_bad_gutenberg_ids_are_rejected_and_valid_urls_use_mirror(self):
        with self.assertRaisesRegex(ValueError, "Mismatched"):
            download_url(
                "https://www.gutenberg.org/cache/epub/57493/pg74739.txt",
                "https://example.org",
            )
        self.assertEqual(
            download_url(
                "https://www.gutenberg.org/cache/epub/74739/pg74739.txt",
                "https://example.org",
            ),
            "https://example.org/74739/pg74739.txt",
        )

    def test_wrappers_removed_without_losing_paragraphs_or_numbers(self):
        body = "An example has 12 parts,\neach with a different purpose. " * 5
        raw = (
            "License header\r\n*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\r\n"
            + body
            + "\n\nA separate paragraph.\n*** END OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\nLicense footer"
        )
        clean = clean_text(raw, require_wrappers=True)
        self.assertNotIn("License", clean)
        self.assertIn("12 parts, each", clean)
        self.assertIn("\n\nA separate paragraph.", clean)
        with self.assertRaises(ValueError):
            clean_text("<html>error</html>", require_wrappers=True)

    def test_seeded_book_splits_are_disjoint(self):
        docs = [{"sha256": str(i)} for i in range(20)]
        first = split_documents([dict(d) for d in docs], 0.1, 0.1, 42)
        second = split_documents([dict(d) for d in docs], 0.1, 0.1, 42)
        self.assertEqual(first, second)
        groups = [
            {d["sha256"] for d in first if d["split"] == split}
            for split in ("train", "validation", "test")
        ]
        self.assertEqual([len(g) for g in groups], [16, 2, 2])
        self.assertFalse(
            groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2]
        )

    def test_packed_blocks_preserve_every_target_and_mask_padding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            values = np.arange(11, dtype="<u2")
            values.tofile(root / "test.bin")
            write_json(
                root / "metadata.json",
                {"dtype": "<u2", "token_counts": {"test": len(values)}},
            )
            blocks = TokenBlocks(root, "test", 4, 99)
            targets = []
            for i in range(len(blocks)):
                row = blocks[i]
                targets.extend(row["labels"][1:][row["labels"][1:] != -100].tolist())
                self.assertTrue(
                    torch.all(row["labels"][row["attention_mask"] == 0] == -100)
                )
            self.assertEqual(targets, values[1:].tolist())

    def test_default_model_has_tied_embeddings_and_about_ten_million_parameters(self):
        class Vocabulary:
            pad_token_id, bos_token_id, eos_token_id = 0, 1, 2

            def __len__(self):
                return 8192

        args = parser().parse_args(["train"])
        model = build_model(Vocabulary(), args)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 10101760)
        self.assertIs(model.lm_head.weight, model.transformer.wte.weight)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.data = cls.root / "prepared"
        sources = []
        for index, topic in enumerate(
            ["gardens", "stars", "rocks", "boats", "birds", "trees", "clouds", "tools"]
        ):
            path = cls.root / f"book-{index}.txt"
            text = (
                f"A short book about {topic}.\n\n"
                + (
                    f"This chapter describes {topic} and their place in the world. "
                    "An observation can be written in a clear sentence. A definition explains a word. "
                    "We compare examples, record details, and consider the explanation.\n\n"
                )
                * 12
            )
            path.write_text(text, encoding="utf-8")
            sources.append(str(path))
        duplicate = cls.root / "duplicate.txt"
        duplicate.write_text(Path(sources[0]).read_text())
        sources.append(str(duplicate))
        cls.corpus = cls.root / "corpus.py"
        cls.corpus.write_text("sources = " + repr(sources))
        cls.prompts = cls.root / "prompts.json"
        cls.prompts.write_text(json.dumps(["A definition", "The stars"]))
        args = parser().parse_args(
            [
                "prepare",
                "--corpus",
                str(cls.corpus),
                "--data-dir",
                str(cls.data),
                "--cache-dir",
                str(cls.root / "cache"),
                "--vocab-size",
                "512",
            ]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(args)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def training_args(self, output, resume=None):
        values = [
            "train",
            "--data-dir",
            str(self.data),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--threads",
            "1",
            "--max-steps",
            "101",
            "--width",
            "32",
            "--layers",
            "1",
            "--heads",
            "2",
            "--context",
            "32",
            "--batch-size",
            "2",
            "--gradient-accumulation",
            "2",
            "--warmup-steps",
            "5",
            "--eval-every",
            "100",
            "--eval-batches",
            "1",
            "--save-every",
            "100",
            "--sample-prompts",
            str(self.prompts),
            "--sample-max-new-tokens",
            "5",
        ]
        if resume:
            values.extend(["--resume", str(resume)])
        return parser().parse_args(values)

    def test_prepare_uses_distinct_books_and_byte_tokenizer(self):
        metadata = json.loads((self.data / "metadata.json").read_text())
        self.assertEqual(len(metadata["documents"]), 8)
        self.assertEqual(len(metadata["duplicate_sources"]), 1)
        self.assertEqual(
            {d["split"] for d in metadata["documents"]}, {"train", "validation", "test"}
        )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            self.data / "tokenizer", local_files_only=True
        )
        unseen = "Unseen text: π, café, and 123."
        self.assertEqual(
            tokenizer.decode(tokenizer.encode(unseen, add_special_tokens=False)), unseen
        )
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare(parser().parse_args(["prepare", "--data-dir", str(self.data)]))

    def test_101_steps_sample_at_100_resume_exactly_and_evaluate(self):
        output = self.root / "run"
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            train(self.training_args(output))
        self.assertIn("Sample responses at step 100", captured.getvalue())
        records = [
            json.loads(line)
            for line in (output / "samples.jsonl").read_text().splitlines()
        ]
        self.assertEqual([r["step"] for r in records], [100])
        self.assertEqual(len(records[0]["samples"]), 2)
        self.assertTrue((output / "checkpoint-100" / "training_state.pt").exists())
        self.assertTrue((output / "best" / "config.json").exists())
        resumed = self.root / "resumed"
        with contextlib.redirect_stdout(io.StringIO()):
            train(self.training_args(resumed, output / "checkpoint-100"))
        self.assertTrue((resumed / "best" / "config.json").exists())
        original = GPT2LMHeadModel.from_pretrained(
            output / "model", local_files_only=True
        )
        restarted = GPT2LMHeadModel.from_pretrained(
            resumed / "model", local_files_only=True
        )
        for name, value in original.state_dict().items():
            torch.testing.assert_close(
                value, restarted.state_dict()[name], rtol=0, atol=0
            )
        report = self.root / "evaluation.json"
        with contextlib.redirect_stdout(io.StringIO()):
            main(
                [
                    "test",
                    "--model-dir",
                    str(output / "model"),
                    "--data-dir",
                    str(self.data),
                    "--device",
                    "cpu",
                    "--threads",
                    "1",
                    "--batch-size",
                    "2",
                    "--sample-prompts",
                    str(self.prompts),
                    "--max-new-tokens",
                    "5",
                    "--report",
                    str(report),
                ]
            )
        result = json.loads(report.read_text())
        self.assertEqual(result["split"], "test")
        self.assertTrue(result["complete_split"])
        self.assertTrue(np.isfinite(result["perplexity"]))
        self.assertGreater(result["target_tokens"], 0)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            output / "model", local_files_only=True
        )
        original.train()
        state = torch.get_rng_state().clone()
        samples = sample_responses(
            original, tokenizer, ["A definition"], torch.device("cpu"), 5
        )
        self.assertTrue(original.training)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertIn("continuation", samples[0])

    def test_failed_sources_are_reported_without_publishing_partial_data(self):
        broken = self.root / "broken.py"
        broken.write_text("sources = ['missing.txt']")
        output = self.root / "failed-output"
        cache = self.root / "failed-cache"
        with (
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(ValueError, "sources failed"),
        ):
            prepare(
                parser().parse_args(
                    [
                        "prepare",
                        "--corpus",
                        str(broken),
                        "--data-dir",
                        str(output),
                        "--cache-dir",
                        str(cache),
                    ]
                )
            )
        self.assertFalse(output.exists())
        self.assertEqual(
            len(
                json.loads((cache / "last-prepare-report.json").read_text())["failures"]
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
