"""Verify reloadable exports, tokenizer preservation, and explicit upload scope."""

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from export_model import (
    download_files,
    export_model,
    upload_export,
    validate_checkpoint,
)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "checkpoint"
        backend = Tokenizer(models.BPE())
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        backend.train_from_iterator(
            ["A fact about stars. Water becomes steam. A clear explanation."],
            trainers.BpeTrainer(
                vocab_size=320,
                show_progress=False,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                special_tokens=["<|pad|>", "<|bos|>", "<|endoftext|>"],
            ),
        )
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="<|pad|>",
            bos_token="<|bos|>",
            eos_token="<|endoftext|>",
        )
        torch.manual_seed(7)
        self.model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(self.tokenizer),
                n_positions=32,
                n_ctx=32,
                n_embd=32,
                n_head=2,
                n_layer=1,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
        )
        self.model.save_pretrained(self.source)
        self.tokenizer.save_pretrained(self.source)
        (self.source / "training_state.pt").write_text("optimizer must stay local")

    def local_export(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return export_model(self.source, self.root / "export", hf_only=True)

    def enable_bos(self):
        self.tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single="<|bos|> $A",
            pair="<|bos|> $A $B",
            special_tokens=[("<|bos|>", self.tokenizer.bos_token_id)],
        )
        self.tokenizer.save_pretrained(self.source)

    def test_bos_prompt_format_survives_hf_export(self):
        self.enable_bos()
        result = self.local_export()
        tokenizer = PreTrainedTokenizerFast.from_pretrained(result["huggingface"])
        expected = [self.tokenizer.bos_token_id] + self.tokenizer.encode(
            "A fact", add_special_tokens=False
        )
        self.assertEqual(tokenizer.encode("A fact"), expected)
        configuration = json.loads(
            (Path(result["huggingface"]) / "tokenizer_config.json").read_text()
        )
        self.assertTrue(configuration["add_bos_token"])
        self.assertFalse(configuration["add_eos_token"])

    def test_extra_prompt_postprocessing_is_rejected(self):
        self.enable_bos()
        path = self.source / "tokenizer.json"
        data = json.loads(path.read_text())
        data["post_processor"]["single"].append(
            {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}}
        )
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "post-processing"):
            validate_checkpoint(self.source)

    def test_hf_round_trip_preserves_weights_and_tokenization(self):
        result = self.local_export()
        reloaded = GPT2LMHeadModel.from_pretrained(
            result["huggingface"], local_files_only=True
        )
        for name, tensor in self.model.state_dict().items():
            torch.testing.assert_close(
                tensor, reloaded.state_dict()[name], rtol=0, atol=0
            )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            result["huggingface"], local_files_only=True
        )
        for prompt in [
            "A fact",
            "  Q. What is steam?\nA.",
            "café—αβ 水 🚀",
            "<|bos|>A fact.<|endoftext|>",
        ]:
            self.assertEqual(self.tokenizer.encode(prompt), tokenizer.encode(prompt))
        with zipfile.ZipFile(result["zip"]) as archive:
            self.assertIn("model.safetensors", archive.namelist())
            self.assertNotIn("training_state.pt", archive.namelist())
        self.assertTrue((self.source / "training_state.pt").exists())
        with self.assertRaisesRegex(ValueError, "new output"):
            export_model(self.source, result["directory"], hf_only=True)

    def test_incompatible_tokenizer_is_rejected_before_export(self):
        path = self.source / "tokenizer.json"
        data = json.loads(path.read_text())
        data["pre_tokenizer"]["add_prefix_space"] = True
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "ByteLevel"):
            validate_checkpoint(self.source)

    def test_upload_is_explicit_and_excludes_unrelated_files(self):
        with patch("huggingface_hub.HfApi") as api_class:
            result = self.local_export()
            api_class.assert_not_called()
            api = api_class.return_value
            api.model_info.return_value.private = True
            (Path(result["huggingface"]) / "training_state.pt").write_text("exclude")
            (Path(result["huggingface"]) / ".env").write_text("exclude")
            with contextlib.redirect_stdout(io.StringIO()):
                upload_export(result["directory"], "example/tidbit")
            api.create_repo.assert_called_once_with(
                repo_id="example/tidbit",
                repo_type="model",
                private=True,
                exist_ok=True,
            )
            paths = {
                op.path_in_repo
                for op in api.create_commit.call_args.kwargs["operations"]
            }
            self.assertIn("model.safetensors", paths)
            self.assertNotIn("training_state.pt", paths)
            self.assertNotIn(".env", paths)
            api.create_commit.reset_mock()
            api.model_info.return_value.private = False
            with self.assertRaisesRegex(ValueError, "repository is public"):
                upload_export(result["directory"], "example/tidbit")
            api.create_commit.assert_not_called()

    def test_colab_download_uses_the_exported_file(self):
        result = self.local_export()
        colab = ModuleType("google.colab")
        colab.files = Mock()
        with patch.dict("sys.modules", {"google.colab": colab}):
            download_files(result["zip"])
        colab.files.download.assert_called_once_with(result["zip"])

    @unittest.skipUnless(
        os.environ.get("TIDBIT_LLAMA_CPP_DIR"),
        "Set TIDBIT_LLAMA_CPP_DIR for GGUF integration",
    )
    def test_gguf_tokenizer_matches_hugging_face(self):
        checkout = Path(os.environ["TIDBIT_LLAMA_CPP_DIR"])
        for bos in (False, True):
            if bos:
                self.enable_bos()
            with contextlib.redirect_stdout(io.StringIO()):
                result = export_model(
                    self.source,
                    self.root / f"gguf-export-{bos}",
                    llama_cpp_dir=checkout,
                )
            for prompt in [
                "A fact",
                "  Q. Steam?\nA.",
                "123.45 café—αβ 水 🚀",
                "<|bos|>A fact.<|endoftext|>",
                "",
            ]:
                output = subprocess.check_output(
                    [
                        str(checkout / "build/bin/llama-tokenize"),
                        "-m",
                        result["gguf"],
                        "--ids",
                        "--stdin",
                        "--log-disable",
                    ],
                    input=prompt,
                    text=True,
                )
                self.assertEqual(self.tokenizer.encode(prompt), json.loads(output))


if __name__ == "__main__":
    unittest.main()
