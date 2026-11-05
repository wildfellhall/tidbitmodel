"""Exercise the standalone file as one notebook cell, without sibling modules."""

import ast
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]


def literal_assignment(tree, name):
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    )


class OnePieceTests(unittest.TestCase):
    def test_corpus_is_embedded_and_imports_are_independent(self):
        script = ast.parse((ROOT / "one_piece.py").read_text())
        corpus = ast.parse((ROOT / "corpus.py").read_text())
        self.assertEqual(
            ast.literal_eval(literal_assignment(script, "sources").value),
            ast.literal_eval(literal_assignment(corpus, "sources").value),
        )
        for node in ast.walk(script):
            if isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0, "Standalone file has a relative import")
                self.assertNotIn(
                    node.module.split(".")[0], {"tidbit", "corpus", "main"}
                )
            elif isinstance(node, ast.Import):
                self.assertFalse(
                    {name.name.split(".")[0] for name in node.names}
                    & {"tidbit", "corpus", "main"}
                )

    def test_single_cell_training_input_reuse_and_automatic_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            isolated = root / "isolated"
            isolated.mkdir()
            copied = isolated / "one_piece.py"
            copied.write_text((ROOT / "one_piece.py").read_text(), encoding="utf-8")
            books = []
            for index, topic in enumerate(
                ["stars", "plants", "birds", "rocks", "clouds", "boats"]
            ):
                path = root / f"book-{index}.txt"
                path.write_text(
                    f"A book about {topic}.\n\n"
                    + (
                        f"We observe {topic} and describe their place in the world. "
                        "A clear explanation uses simple words and examples. "
                        "A definition explains a word, and a question invites a response.\n\n"
                    )
                    * 10,
                    encoding="utf-8",
                )
                books.append(str(path))
            tree = ast.parse(copied.read_text())
            configuration = ast.literal_eval(literal_assignment(tree, "CONFIG").value)
            configuration.update(
                {
                    "data_dir": str(root / "data"),
                    "cache_dir": str(root / "cache"),
                    "output_dir": str(root / "run"),
                    "device": "cpu",
                    "threads": 1,
                    "vocab_size": 512,
                    "max_steps": 101,
                    "batch_size": 2,
                    "gradient_accumulation": 2,
                    "context": 32,
                    "width": 32,
                    "heads": 2,
                    "layers": 1,
                    "warmup_steps": 5,
                    "eval_every": 100,
                    "save_every": 100,
                    "eval_batches": 1,
                    "sample_prompts": ["A definition", "The stars"],
                    "sample_max_new_tokens": 5,
                    "max_new_tokens": 5,
                    "max_batches": 1,
                }
            )
            literal_assignment(tree, "CONFIG").value = ast.parse(
                repr(configuration), mode="eval"
            ).body
            literal_assignment(tree, "sources").value = ast.parse(
                repr(books), mode="eval"
            ).body
            literal_assignment(
                tree, "INSTALL_MISSING_DEPENDENCIES"
            ).value = ast.Constant(False)
            code = compile(
                ast.fix_missing_locations(tree), "<single notebook cell>", "exec"
            )
            namespace = {"__name__": "__main__"}
            captured = io.StringIO()
            before = Path.cwd()
            try:
                os.chdir(isolated)
                with (
                    patch.object(
                        sys, "argv", ["ipykernel_launcher", "-f", "kernel.json"]
                    ),
                    patch(
                        "builtins.input",
                        side_effect=["", "A clear explanation", "/quit"],
                    ),
                    patch(
                        "subprocess.check_call",
                        side_effect=AssertionError("Tests must not install packages"),
                    ),
                    contextlib.redirect_stdout(captured),
                ):
                    # Execute trusted repository code as a pasted notebook cell.
                    exec(code, namespace)  # noqa: S102
            finally:
                os.chdir(before)
            self.assertNotIn("__file__", namespace)
            self.assertIn("Sample responses at step 100", captured.getvalue())
            self.assertIn("Interactive testing is ready", captured.getvalue())
            self.assertIn("Response:", captured.getvalue())
            self.assertIn("Interactive testing finished.", captured.getvalue())
            output = root / "run"
            samples = [
                json.loads(line)
                for line in (output / "samples.jsonl").read_text().splitlines()
            ]
            self.assertEqual([sample["step"] for sample in samples], [100])
            report = json.loads((output / "test-results.json").read_text())
            self.assertGreater(report["target_tokens"], 0)

            # A second execution reuses the corpus and final model, then accepts prompts.
            again = io.StringIO()
            with contextlib.redirect_stdout(again):
                namespace["run"](input_fn=lambda _: "/quit")
            self.assertIn("Reusing prepared corpus", again.getvalue())
            self.assertIn("Reusing completed training run", again.getvalue())
            self.assertEqual(
                len((output / "samples.jsonl").read_text().splitlines()), 1
            )
            with self.assertRaisesRegex(ValueError, "different sources/settings"):
                namespace["run"]({"vocab_size": 513}, input_fn=lambda _: "/quit")
            with self.assertRaisesRegex(ValueError, "different training settings"):
                namespace["run"]({"width": 64}, input_fn=lambda _: "/quit")

            # Simulate interruption after checkpoint 100, before the final model exists.
            original_path = root / "original-final"
            (output / "model").rename(original_path)
            (output / "best").rename(root / "original-best")
            resumed_log = io.StringIO()
            with contextlib.redirect_stdout(resumed_log):
                namespace["run"](input_fn=lambda _: "/quit")
            self.assertIn("Resuming training from", resumed_log.getvalue())
            original = GPT2LMHeadModel.from_pretrained(
                original_path, local_files_only=True
            )
            resumed = GPT2LMHeadModel.from_pretrained(
                output / "model", local_files_only=True
            )
            for name, value in original.state_dict().items():
                torch.testing.assert_close(
                    value, resumed.state_dict()[name], rtol=0, atol=0
                )

            # Completion can be checkpointed before the final model is exported.
            last_checkpoint = output / "checkpoint-101"
            (output / "model").rename(last_checkpoint)
            (output / "best").rename(root / "completed-best")
            completed_log = io.StringIO()
            with contextlib.redirect_stdout(completed_log):
                selected = namespace["run"](input_fn=lambda _: "/quit")
            self.assertEqual(selected, last_checkpoint)
            self.assertIn("Reusing completed checkpoint", completed_log.getvalue())


if __name__ == "__main__":
    unittest.main()
