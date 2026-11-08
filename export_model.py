#!/usr/bin/env python3
"""Export a trained Tidbit checkpoint independently of the training cell.

    python -m pip install -r requirements-export.txt
    python export_model.py export --model-dir runs/one_piece-coherent/best
    python export_model.py export --model-dir runs/tidbit-coherent/best --outtype q8_0
    python export_model.py upload --export-dir runs/exports/tidbit --repo-id USER/tidbit

Export creates a Hugging Face folder, its ZIP, and a GGUF in a NEW output
directory. F16 is the default; Q8_0 is optional. --download starts a Colab
browser download or displays download links in Jupyter. On a local computer,
the exported files are already on disk. --hf-only skips GGUF conversion.

Notebook use (after installing requirements-export.txt):
    from export_model import export_model, download_files, upload_export
    result = export_model("runs/one_piece-coherent/best")
    download_files(result["gguf"])
    # Only run this if you want to upload; authenticate using hf auth login
    # or HF_TOKEN first. New Hub repositories are private by default.
    # upload_export(result["directory"], "USER/tidbit")

GGUF conversion fetches a pinned llama.cpp checkout using Git on first use.
It uses the official GPT-2 tensor converter with a validated ByteLevel BPE
adapter: a tokenizer trained from scratch cannot match the converter's
hashes of pretrained vocabularies. No pretrained weights are downloaded.
Training optimizer/RNG files and corpus data are excluded from exports.
The checkpoint, training files, and repository README are never modified.
"""

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LLAMA_CPP_REVISION = "7d7f0287530bc8981690f361d8c1627dbae1c99a"
LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp.git"
DEFAULT_OUTPUT = "runs/exports/tidbit"
DEFAULT_TOOLS = "runs/export-tools/llama.cpp"


def validate_checkpoint(model_dir):
    """Reject architectures/tokenizers the narrow GGUF adapter cannot preserve."""
    model_dir = Path(model_dir).expanduser().resolve()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["GPT2LMHeadModel"]:
        raise ValueError("Expected a saved GPT2LMHeadModel checkpoint.")
    width = config["n_embd"]
    if (
        config.get("n_inner") not in (None, 4 * width)
        or config.get("activation_function", "gelu_new") != "gelu_new"
        or config.get("add_cross_attention", False)
        or config.get("scale_attn_by_inverse_layer_idx", False)
        or not config.get("scale_attn_weights", True)
        or config.get("n_ctx", config["n_positions"]) != config["n_positions"]
    ):
        raise ValueError(
            "This exporter supports the repository's standard GPT-2 architecture."
        )
    pre = tokenizer.get("pre_tokenizer") or {}
    bpe = tokenizer.get("model") or {}
    if (
        tokenizer.get("normalizer") is not None
        or pre.get("type") != "ByteLevel"
        or pre.get("add_prefix_space") is not False
        or pre.get("use_regex") is not True
        or (tokenizer.get("decoder") or {}).get("type") != "ByteLevel"
        or bpe.get("type") != "BPE"
        or any(
            bpe.get(k)
            for k in (
                "dropout",
                "unk_token",
                "continuing_subword_prefix",
                "end_of_word_suffix",
                "byte_fallback",
                "ignore_merges",
            )
        )
    ):
        raise ValueError(
            "GGUF requires the unmodified ByteLevel BPE tokenizer used by Tidbit."
        )
    post = tokenizer.get("post_processor")
    sequence = {"Sequence": {"id": "A", "type_id": 0}}
    bos = {"SpecialToken": {"id": "<|bos|>", "type_id": 0}}
    if post is not None:
        if post.get("type") != "TemplateProcessing" or post.get("single") not in (
            [sequence],
            [bos, sequence],
        ):
            raise ValueError(
                "Tokenizer post-processing must preserve text with at most a leading BOS."
            )
        if post["single"] == [bos, sequence] and post.get("special_tokens", {}).get(
            "<|bos|>"
        ) != {
            "id": "<|bos|>",
            "ids": [config["bos_token_id"]],
            "tokens": ["<|bos|>"],
        }:
            raise ValueError("Tokenizer post-processing has a mismatched BOS token.")
    vocab = bpe["vocab"]
    if sorted(vocab.values()) != list(range(config["vocab_size"])):
        raise ValueError("Tokenizer vocabulary IDs must match the model's vocabulary.")
    from tokenizers.pre_tokenizers import ByteLevel

    if not set(ByteLevel.alphabet()).issubset(vocab):
        raise ValueError("The tokenizer must preserve all 256 byte symbols.")
    special_tokens = {"<|bos|>", "<|endoftext|>", "<|pad|>"}
    for token in tokenizer.get("added_tokens", []):
        if (
            token.get("content") not in special_tokens
            or not token.get("special")
            or any(
                token.get(key)
                for key in ("single_word", "lstrip", "rstrip", "normalized")
            )
        ):
            raise ValueError(
                "Modified added-token behavior is not supported by the GGUF adapter."
            )
    for name, token in (
        ("bos", "<|bos|>"),
        ("eos", "<|endoftext|>"),
        ("pad", "<|pad|>"),
    ):
        if vocab.get(token) != config.get(f"{name}_token_id") or token not in vocab:
            raise ValueError(f"Mismatched {name} token in model and tokenizer.")
    if not list(model_dir.glob("*.safetensors")):
        raise ValueError(
            "Expected the safetensors checkpoint saved by Tidbit training."
        )
    return model_dir, config


def ensure_converter(path):
    """Fetch a reproducible converter; never alter an existing checkout."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".llama-", dir=path.parent) as tmp:
            checkout = Path(tmp) / "checkout"
            subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "fetch",
                    "--quiet",
                    "--depth",
                    "1",
                    LLAMA_CPP_URL,
                    LLAMA_CPP_REVISION,
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "checkout",
                    "--quiet",
                    "--detach",
                    "FETCH_HEAD",
                ],
                check=True,
            )
            checkout.rename(path)
    revision = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if revision != LLAMA_CPP_REVISION or dirty:
        raise ValueError(
            "Converter checkout differs from the tested revision; use a new --llama-cpp-dir."
        )
    if not (path / "conversion/gpt2.py").is_file():
        raise ValueError("The llama.cpp GPT-2 converter is missing.")
    return path


def convert_gguf(model_dir, destination, outtype, llama_cpp_dir):
    """Use upstream tensor mapping and preserve our custom tokenizer IDs/merges."""
    model_dir, config = validate_checkpoint(model_dir)
    checkout = ensure_converter(llama_cpp_dir)
    # Import the GGUF library shipped with the pinned converter, not a different
    # system-installed version. Conversion runs in a fresh subprocess below.
    sys.path[:0] = [str(checkout), str(checkout / "gguf-py")]
    import gguf
    from conversion.gpt2 import GPT2Model

    class TidbitGPT2Model(GPT2Model):
        model_arch = gguf.MODEL_ARCH.GPT2

        def get_vocab_base_pre(self, tokenizer):
            # The JSON structure was validated above. Vocab-dependent hashes
            # cannot identify a newly trained vocabulary even with GPT-2 regex.
            return "gpt-2"

    file_types = {
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "f32": gguf.LlamaFileType.ALL_F32,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    }
    config.setdefault("n_ctx", config["n_positions"])
    converter = TidbitGPT2Model(
        model_dir,
        file_types[outtype],
        Path(destination),
        hparams=config,
        model_name="Tidbit",
        eager=True,
    )
    converter.write()
    # Check that the result is readable, populated, and has the intended vocab.
    reader = gguf.GGUFReader(str(destination))
    if (
        not reader.tensors
        or len(reader.get_field("tokenizer.ggml.tokens").data) != config["vocab_size"]
    ):
        raise ValueError(
            "GGUF validation failed: missing tensors or wrong vocabulary size."
        )


def export_model(
    model_dir=None,
    output_dir=DEFAULT_OUTPUT,
    outtype="f16",
    llama_cpp_dir=DEFAULT_TOOLS,
    hf_only=False,
):
    """Create local artifacts. This function never uploads to the Hub."""
    if outtype not in {"f16", "f32", "q8_0"}:
        raise ValueError("outtype must be f16, f32, or q8_0")
    if model_dir is None:
        candidates = [
            Path("runs") / run / kind
            for run in ("one_piece-coherent", "tidbit-coherent", "one_piece", "tidbit")
            for kind in ("best", "model")
        ]
        model_dir = next((p for p in candidates if (p / "config.json").is_file()), None)
        if model_dir is None:
            raise ValueError(
                "No trained model found. Supply --model-dir pointing to best or model."
            )
    model_dir, config = validate_checkpoint(model_dir)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() or output == model_dir or model_dir in output.parents:
        raise ValueError("Choose a new output directory outside the source checkpoint.")
    output.parent.mkdir(parents=True, exist_ok=True)
    from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

    with tempfile.TemporaryDirectory(
        prefix=".tidbit-export-", dir=output.parent
    ) as tmp:
        staging = Path(tmp) / "artifacts"
        hf = staging / "huggingface"
        model = GPT2LMHeadModel.from_pretrained(model_dir, local_files_only=True)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_dir, local_files_only=True
        )
        model.save_pretrained(hf, safe_serialization=True)
        tokenizer.save_pretrained(hf)
        # Match the saved HF post-processor when llama.cpp constructs prompts.
        token_config_path = hf / "tokenizer_config.json"
        token_config = json.loads(token_config_path.read_text(encoding="utf-8"))
        post = json.loads((hf / "tokenizer.json").read_text()).get("post_processor")
        add_bos = bool(post and len(post["single"]) == 2)
        token_config.update(add_bos_token=add_bos, add_eos_token=False)
        token_config_path.write_text(
            json.dumps(token_config, indent=2) + "\n", encoding="utf-8"
        )
        parameters = model.num_parameters()
        del model, tokenizer
        gguf_name = None
        if not hf_only:
            gguf_name = f"tidbit-{outtype}.gguf"
            # A fresh interpreter isolates converter imports from notebook state.
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "_convert",
                    "--model-dir",
                    str(hf),
                    "--output",
                    str(staging / gguf_name),
                    "--outtype",
                    outtype,
                    "--llama-cpp-dir",
                    str(Path(llama_cpp_dir).resolve()),
                ],
                check=True,
            )
        shutil.make_archive(str(staging / "tidbit-huggingface"), "zip", root_dir=hf)
        manifest = {
            "parameters": parameters,
            "vocab_size": config["vocab_size"],
            "context": config["n_positions"],
            "gguf": gguf_name,
            "gguf_type": None if hf_only else outtype,
            "llama_cpp_revision": None if hf_only else LLAMA_CPP_REVISION,
            "files": {
                str(p.relative_to(staging)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in staging.rglob("*")
                if p.is_file()
            },
        }
        (staging / "export.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    result = {
        "directory": str(output),
        "huggingface": str(output / "huggingface"),
        "zip": str(output / "tidbit-huggingface.zip"),
        "gguf": str(output / gguf_name) if gguf_name else None,
    }
    print(json.dumps(result, indent=2))
    return result


def upload_export(export_dir, repo_id, public=False):
    """Explicitly upload only inference artifacts, using saved Hugging Face auth."""
    from huggingface_hub import CommitOperationAdd, HfApi

    root = Path(export_dir).expanduser().resolve()
    manifest = json.loads((root / "export.json").read_text(encoding="utf-8"))
    # Fixed allowlist prevents optimizer states, credentials, or unrelated files
    # added to the export directory from accidentally entering the upload.
    allowed = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
    }
    files = [
        p
        for p in (root / "huggingface").iterdir()
        if p.is_file()
        and (
            p.name in allowed
            or p.name == "model.safetensors"
            or (p.name.startswith("model-") and p.suffix == ".safetensors")
        )
    ]
    validate_checkpoint(root / "huggingface")
    operations = [
        CommitOperationAdd(path_in_repo=p.name, path_or_fileobj=str(p)) for p in files
    ]
    if manifest.get("gguf"):
        name = manifest["gguf"]
        if Path(name).name != name or not name.endswith(".gguf"):
            raise ValueError("Invalid GGUF path in export manifest.")
        operations.append(
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(root / name))
        )
    api = HfApi()
    # create_repo does not change the visibility of an existing repository.
    api.create_repo(
        repo_id=repo_id, repo_type="model", private=not public, exist_ok=True
    )
    if not public and not api.model_info(repo_id).private:
        raise ValueError(
            "That repository is public. Use --public or choose a new private repository."
        )
    result = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message="Upload Tidbit model and inference artifacts",
    )
    print(f"Uploaded: https://huggingface.co/{repo_id}")
    return result


def download_files(*paths):
    """Download from Colab, link from Jupyter, or show existing local files."""
    paths = [Path(p).resolve() for p in paths if p is not None]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("An exported file is missing.")
    try:
        from google.colab import files
    except ImportError:
        try:
            from IPython import get_ipython
            from IPython.display import FileLink, display

            if get_ipython() is not None:
                for path in paths:
                    display(FileLink(str(path.relative_to(Path.cwd()))))
                return
        except (ImportError, ValueError):
            pass
        for path in paths:
            print(f"Saved locally: {path}")
    else:
        for path in paths:
            files.download(str(path))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_convert":
        worker = argparse.ArgumentParser()
        worker.add_argument("--model-dir", required=True)
        worker.add_argument("--output", required=True)
        worker.add_argument("--outtype", choices=["f16", "f32", "q8_0"], required=True)
        worker.add_argument("--llama-cpp-dir", required=True)
        args = worker.parse_args(sys.argv[2:])
        logging.basicConfig(level=logging.INFO)
        convert_gguf(args.model_dir, args.output, args.outtype, args.llama_cpp_dir)
        return
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = cli.add_subparsers(dest="command", required=True)
    export = commands.add_parser(
        "export", help="Create GGUF and Hugging Face files locally"
    )
    export.add_argument("--model-dir")
    export.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    export.add_argument("--outtype", choices=["f16", "f32", "q8_0"], default="f16")
    export.add_argument("--llama-cpp-dir", default=DEFAULT_TOOLS)
    export.add_argument("--hf-only", action="store_true")
    export.add_argument("--download", action="store_true")
    upload = commands.add_parser(
        "upload", help="Explicitly upload an existing export to Hugging Face"
    )
    upload.add_argument("--export-dir", default=DEFAULT_OUTPUT)
    upload.add_argument("--repo-id", required=True)
    upload.add_argument(
        "--public", action="store_true", help="Allow upload to a public repository"
    )
    args = cli.parse_args()
    if args.command == "export":
        result = export_model(
            args.model_dir,
            args.output_dir,
            args.outtype,
            args.llama_cpp_dir,
            args.hf_only,
        )
        if args.download:
            download_files(result["gguf"] or result["zip"])
    elif args.command == "upload":
        upload_export(args.export_dir, args.repo_id, args.public)


if __name__ == "__main__":
    main()
