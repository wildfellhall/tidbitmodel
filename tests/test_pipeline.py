"""Offline regression and end-to-end tests: python -m unittest discover -s tests -v."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from main import main, parser
from tidbit.data import (
    TokenBlocks,
    clean_text,
    document_identity,
    download_url,
    fetch_source,
    load_sources,
    passage_tokens,
    prepare,
    prose_paragraphs,
    split_documents,
    write_json,
)
from tidbit.training import (
    ShuffledEpochSampler,
    build_model,
    sample_responses,
    train,
    update_parameters,
)


class ParsingTests(unittest.TestCase):
    def test_prose_filter_keeps_question_answer_pairs_and_removes_numeric_lists(self):
        text = "Q. What is steam?\n\nA. It is water in its gaseous state.\n\n12 43 99 100 78.\n\nCHAPTER I"
        kept = list(prose_paragraphs(text))
        self.assertEqual(len(kept), 1)
        self.assertIn("Q. What is steam?\nA.", kept[0])

    def test_work_families_are_not_split_across_train_validation_test(self):
        docs = [{"sha256": str(i), "group": str(i // 2)} for i in range(20)]
        assigned = split_documents(docs, 0.1, 0.1, 42)
        for group in {d["group"] for d in assigned}:
            self.assertEqual(
                len({d["split"] for d in assigned if d["group"] == group}), 1
            )

    def test_sampler_covers_each_epoch_once_and_resumes_mid_epoch(self):
        lengths = np.arange(11) + 2
        sampler = ShuffledEpochSampler(11, 42, lengths, batch_size=4)
        first = sampler.take(4)
        resumed = ShuffledEpochSampler(11, 7, lengths, batch_size=4)
        resumed.load_state_dict(sampler.state_dict())
        for _ in range(2):
            more = sampler.take(4)
            self.assertEqual(more, resumed.take(4))
            first.extend(more)
        self.assertEqual(sorted(first), list(range(11)))
        self.assertEqual(sampler.progress, 1)
        second = []
        for _ in range(3):
            more = sampler.take(4)
            self.assertEqual(more, resumed.take(4))
            second.extend(more)
        self.assertEqual(sorted(second), list(range(11)))
        self.assertEqual(sampler.progress, 2)

    def test_volume_titles_share_a_work_family(self):
        first = document_identity(
            "Title: Roman History, Books I-III\nAuthor: An Author\n*** START",
            "fallback",
        )
        second = document_identity(
            "Title: Roman History, Books IV-VI\nAuthor: An Author\n*** START",
            "fallback",
        )
        unrelated = document_identity(
            "Title: Other History\nAuthor: An Author\n*** START", "fallback"
        )
        self.assertEqual(first[1], second[1])
        self.assertNotEqual(first[1], unrelated[1])

    def test_missing_generated_text_uses_a_validated_legacy_mirror_file(self):
        session = MagicMock()
        missing, found = MagicMock(), MagicMock()
        missing.__enter__.return_value = missing
        found.__enter__.return_value = found
        missing.status_code, found.status_code = 404, 200
        body = (
            "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            + "A clear sentence about stars. " * 20
            + "\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***"
        )
        found.iter_content.return_value = [body.encode()]
        session.get.side_effect = [missing, found]
        with (
            tempfile.TemporaryDirectory() as folder,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            destination = Path(folder) / "download.txt"
            fetch_source(
                session,
                "https://www.gutenberg.org/cache/epub/1234/pg1234.txt",
                "https://example.org/gutenberg-epub",
                destination,
                0,
            )
            self.assertEqual(destination.read_text(), body)
            self.assertFalse(destination.with_suffix(".part").exists())
        self.assertEqual(
            session.get.call_args.args[0],
            "https://example.org/gutenberg/1/2/3/1234/1234-0.txt",
        )

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
                "--context",
                "32",
                "--min-document-words",
                "0",
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
        self.assertGreater(metadata["filter_counts"]["duplicate_paragraphs"], 0)
        self.assertEqual(metadata["format_version"], 2)
        self.assertEqual(
            {d["split"] for d in metadata["documents"]}, {"train", "validation", "test"}
        )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            self.data / "tokenizer", local_files_only=True
        )
        unseen = "Unseen text: π, café, and 123."
        self.assertEqual(tokenizer.encode(unseen)[0], tokenizer.bos_token_id)
        self.assertEqual(
            tokenizer.decode(tokenizer.encode(unseen, add_special_tokens=False)), unseen
        )
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare(parser().parse_args(["prepare", "--data-dir", str(self.data)]))

    def test_generation_uses_the_training_bos_even_after_prompt_truncation(self):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.data / "tokenizer")
        model = build_model(tokenizer, self.training_args(self.root / "unused"))
        for prompt in ("", "A definition", "A definition " * 40):
            with patch.object(
                model, "generate", side_effect=lambda **kw: kw["input_ids"]
            ) as generate:
                sample_responses(model, tokenizer, [prompt], torch.device("cpu"), 8, 0)
            ids = generate.call_args.kwargs["input_ids"][0].tolist()
            self.assertEqual(ids[0], tokenizer.bos_token_id)
            self.assertLessEqual(len(ids) + 8, model.config.n_positions)
            self.assertEqual(ids.count(tokenizer.bos_token_id), 1)

    def test_passages_preserve_tokens_and_do_not_teach_eos_at_hard_cuts(self):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.data / "tokenizer")
        text = "An unusually long explanation " * 20 + "ends here. A final sentence."
        rows = list(
            passage_tokens(
                text,
                tokenizer.backend_tokenizer,
                32,
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
            )
        )
        restored = []
        self.assertTrue(any(row[-1] != tokenizer.eos_token_id for row in rows[:-1]))
        for row in rows:
            self.assertLessEqual(len(row), 32)
            self.assertEqual(row[0], tokenizer.bos_token_id)
            restored.extend(row[1:-1] if row[-1] == tokenizer.eos_token_id else row[1:])
        self.assertEqual(restored, tokenizer.encode(text, add_special_tokens=False))
        self.assertEqual(rows[-1][-1], tokenizer.eos_token_id)
        data = TokenBlocks(self.data, "train", 32, tokenizer.pad_token_id)
        targets = sum(
            int((data[i]["labels"][1:] != -100).sum()) for i in range(len(data))
        )
        metadata = json.loads((self.data / "metadata.json").read_text())
        self.assertEqual(targets, metadata["target_counts"]["train"])

    def test_epoch_budget_consumes_exactly_two_passes_including_partial_batches(self):
        output = self.root / "epochs-run"
        args = self.training_args(output)
        args.max_steps, args.epochs, args.early_stopping_patience = None, 2, 0
        with contextlib.redirect_stdout(io.StringIO()):
            train(args)
        summary = json.loads((output / "training_summary.json").read_text())
        metadata = json.loads((self.data / "metadata.json").read_text())
        self.assertEqual(summary["epochs_completed"], 2)
        self.assertEqual(
            summary["target_tokens_seen"], 2 * metadata["target_counts"]["train"]
        )

    def test_early_stopping_saves_a_terminal_checkpoint(self):
        output = self.root / "early-stop"
        args = self.training_args(output)
        args.eval_every, args.early_stopping_patience = 1, 2
        with (
            patch("tidbit.training.evaluate", return_value={"loss": 5.0}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            train(args)
        summary = json.loads((output / "training_summary.json").read_text())
        self.assertTrue(summary["stopped_early"])
        self.assertLess(summary["steps"], args.max_steps)
        self.assertGreaterEqual(
            summary["epochs_completed"], args.early_stopping_min_epochs
        )
        state = torch.load(output / "model/training_state.pt", weights_only=True)
        self.assertTrue(state["completed"])

    def test_skipped_update_retries_without_consuming_passages_or_schedule(self):
        control, retried = self.root / "no-overflow", self.root / "retry-overflow"
        args = self.training_args(control)
        args.max_steps = 2
        args.dropout = 0
        with contextlib.redirect_stdout(io.StringIO()):
            train(args)
        args = self.training_args(retried)
        args.max_steps = 2
        args.dropout = 0
        attempts = []

        def update(*values):
            attempts.append(1)
            if len(attempts) == 1:
                return torch.tensor(float("inf")), False
            return update_parameters(*values)

        with (
            patch("tidbit.training.update_parameters", side_effect=update),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            train(args)
        self.assertEqual(len(attempts), 3)
        for filename in ("training_summary.json",):
            self.assertEqual(
                json.loads((control / filename).read_text()),
                json.loads((retried / filename).read_text()),
            )
        original = GPT2LMHeadModel.from_pretrained(control / "model")
        resumed = GPT2LMHeadModel.from_pretrained(retried / "model")
        for name, tensor in original.state_dict().items():
            torch.testing.assert_close(
                tensor, resumed.state_dict()[name], rtol=0, atol=0
            )

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA GPU required for native FP16/fused-AdamW integration",
    )
    def test_cuda_fp16_checkpoint_resumes_scaler_and_optimizer(self):
        output = self.root / "cuda-half"
        args = self.training_args(output)
        args.device, args.precision, args.max_steps = "cuda", "float16", 4
        args.save_every, args.eval_every = 2, 2
        with contextlib.redirect_stdout(io.StringIO()):
            train(args)
        state = torch.load(
            output / "checkpoint-2/training_state.pt",
            weights_only=True,
            map_location="cpu",
        )
        self.assertIn("scale", state["scaler"])
        args = self.training_args(self.root / "cuda-resumed", output / "checkpoint-2")
        args.device, args.precision, args.max_steps = "cuda", "float16", 4
        args.save_every, args.eval_every = 2, 2
        with contextlib.redirect_stdout(io.StringIO()):
            train(args)
        final = torch.load(
            self.root / "cuda-resumed/model/training_state.pt",
            weights_only=True,
            map_location="cpu",
        )
        self.assertEqual(final["step"], 4)
        self.assertEqual(final["scaler"]["_growth_tracker"], 4)

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
