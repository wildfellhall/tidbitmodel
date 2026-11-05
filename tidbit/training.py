"""Train GPT2LMHeadModel from random weights; evaluate and sample locally.

Training steps are optimizer updates, including all accumulation microbatches.
Samples are printed and appended to samples.jsonl every 100 optimizer updates.
Validation/test perplexity measures language modeling, not factual accuracy.
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from .data import TokenBlocks, write_json

SAMPLE_EVERY = 100


def choose_device(requested):
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable.")
    return torch.device(requested)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state(device):
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    elif device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def read_prompts(path):
    prompts = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(p, str) for p in prompts)
    ):
        raise ValueError(
            "The sample prompts file must be a nonempty JSON list of strings."
        )
    return prompts


def sample_responses(
    model, tokenizer, prompts, device, max_new_tokens=64, temperature=0.8, seed=42
):
    """Generate without consuming training RNG state or leaving dropout disabled."""
    if max_new_tokens < 1 or temperature <= 0:
        raise ValueError("max-new-tokens and temperature must be positive.")
    previous_mode, previous_rng = model.training, rng_state(device)
    samples = []
    try:
        torch.manual_seed(seed)
        model.eval()
        budget = min(max_new_tokens, model.config.n_positions - 1)
        with torch.inference_mode():
            for prompt in prompts:
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                ids = (ids or [tokenizer.bos_token_id])[
                    -(model.config.n_positions - budget) :
                ]
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=budget,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=min(50, len(tokenizer)),
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bad_words_ids=[[tokenizer.pad_token_id], [tokenizer.bos_token_id]],
                    use_cache=True,
                )
                continuation = tokenizer.decode(
                    output[0, len(ids) :], skip_special_tokens=True
                )
                samples.append({"prompt": prompt, "continuation": continuation})
    finally:
        model.train(previous_mode)
        restore_rng(previous_rng, device)
    return samples


def show_samples(samples, step=None):
    label = f" at step {step}" if step is not None else ""
    print(f"\n--- Sample responses{label} ---", flush=True)
    for sample in samples:
        print(
            f"Prompt: {sample['prompt']}\nResponse: {sample['continuation']}\n",
            flush=True,
        )


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def evaluate(model, dataset, device, batch_size, max_batches=0):
    """Token-weighted loss; padding and the first context token are not targets."""
    previous_mode = model.training
    total_loss, total_targets, batches = 0.0, 0, 0
    # A dedicated generator prevents DataLoader setup from consuming training RNG.
    limit = max_batches * batch_size
    selection = dataset
    if limit and limit < len(dataset):
        # Cover the split instead of evaluating only the opening of its first book.
        selection = Subset(
            dataset, np.linspace(0, len(dataset) - 1, limit, dtype=int).tolist()
        )
    loader = DataLoader(
        selection,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(0),
    )
    try:
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                targets = int((batch["labels"][:, 1:] != -100).sum().item())
                loss = model(**batch, use_cache=False).loss.item()
                if not math.isfinite(loss):
                    raise ValueError("Non-finite evaluation loss.")
                total_loss += loss * targets
                total_targets += targets
                batches += 1
                if max_batches and batches >= max_batches:
                    break
    finally:
        model.train(previous_mode)
    if total_targets == 0:
        raise ValueError("No evaluation targets were found.")
    mean = total_loss / total_targets
    return {
        "loss": mean,
        "perplexity": math.exp(mean) if mean < 700 else None,
        "target_tokens": total_targets,
        "batches": batches,
        "complete_split": len(selection) == len(dataset),
    }


def build_model(tokenizer, args):
    if args.width % args.heads:
        raise ValueError("width must be divisible by heads.")
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.context,
        n_ctx=args.context,
        n_embd=args.width,
        n_layer=args.layers,
        n_head=args.heads,
        n_inner=4 * args.width,
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        tie_word_embeddings=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    # This constructor initializes random weights. No pretrained GPT-2 weights.
    config._attn_implementation = "eager"
    return GPT2LMHeadModel(config)


def save_checkpoint(
    path, model, tokenizer, optimizer, scheduler, sampler, step, best_loss, run, device
):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    write_json(path / "run.json", run)
    temporary = path / "training_state.pt.tmp"
    torch.save(
        {
            "step": step,
            "best_validation_loss": best_loss,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "sampler_rng": sampler.get_state(),
            "rng": rng_state(device),
        },
        temporary,
    )
    temporary.replace(path / "training_state.pt")


def train(args):
    if (
        min(
            args.max_steps,
            args.batch_size,
            args.gradient_accumulation,
            args.layers,
            args.width,
            args.heads,
            args.threads,
            args.eval_every,
            args.save_every,
            args.log_every,
        )
        <= 0
    ):
        raise ValueError(
            "Step intervals, batch sizes, model dimensions, and threads must be positive."
        )
    if (
        args.context < 2
        or not 0 <= args.dropout < 1
        or args.warmup_steps < 0
        or args.eval_batches < 0
    ):
        raise ValueError(
            "Invalid context, dropout, warmup, or evaluation-batch setting."
        )
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay nonnegative.")
    if args.sample_max_new_tokens < 1:
        raise ValueError("sample-max-new-tokens must be positive.")
    prompts = read_prompts(args.sample_prompts)
    output, data = Path(args.output_dir), Path(args.data_dir)
    metadata = json.loads((data / "metadata.json").read_text())
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        data / "tokenizer", local_files_only=True
    )
    device = choose_device(args.device)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    configuration = {
        "context": args.context,
        "width": args.width,
        "layers": args.layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
    }
    run = {
        "dataset_fingerprint": metadata["fingerprint"],
        "configuration": configuration,
        "device": str(device),
        "sample_every": SAMPLE_EVERY,
        "sample_prompts": prompts,
        "sample_max_new_tokens": args.sample_max_new_tokens,
        "torch_version": str(torch.__version__),
    }
    state = None
    if args.resume:
        checkpoint = Path(args.resume)
        previous = json.loads((checkpoint / "run.json").read_text())
        if previous["dataset_fingerprint"] != run["dataset_fingerprint"]:
            raise ValueError("Checkpoint and prepared corpus/tokenizer do not match.")
        if previous["configuration"] != configuration:
            raise ValueError(
                "Resume requires the original training settings, including the planned max-steps."
            )
        model = GPT2LMHeadModel.from_pretrained(
            checkpoint, local_files_only=True, attn_implementation="eager"
        )
        state = torch.load(
            checkpoint / "training_state.pt", map_location="cpu", weights_only=True
        )
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError(
                f"{output} is not empty. Use a new output directory or --resume."
            )
        model = build_model(tokenizer, args)
    if output.exists() and any(output.iterdir()):
        existing = output / "run.json"
        existing_run = json.loads(existing.read_text()) if existing.exists() else {}
        if (
            existing_run.get("dataset_fingerprint") != metadata["fingerprint"]
            or existing_run.get("configuration") != configuration
        ):
            raise ValueError("The output directory belongs to a different run.")
    output.mkdir(parents=True, exist_ok=True)
    model.to(device)
    run["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    write_json(output / "run.json", run)
    training = TokenBlocks(data, "train", args.context, tokenizer.pad_token_id)
    validation = TokenBlocks(data, "validation", args.context, tokenizer.pad_token_id)
    groups = [
        {
            "params": [p for p in model.parameters() if p.ndim >= 2],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate)
    warmup = min(args.warmup_steps, max(0, args.max_steps - 1))

    def learning_rate_factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, args.max_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    sampler = torch.Generator().manual_seed(args.seed + 1)
    start, best_loss = 0, float("inf")
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sampler.set_state(state["sampler_rng"])
        restore_rng(state["rng"], device)
        start, best_loss = state["step"], state["best_validation_loss"]
        # A resumed run in a new directory needs its own best checkpoint.
        # The previous best model may belong to an earlier, unavailable step.
        if not (output / "best" / "training_state.pt").exists():
            best_loss = float("inf")
    if start >= args.max_steps:
        raise ValueError("This checkpoint already reached the planned max-steps.")
    print(
        f"Training {run['parameters']:,} randomly initialized parameters on {device}; samples every {SAMPLE_EVERY} optimizer steps.",
        flush=True,
    )
    print(
        f"Train tokens: {len(training.tokens):,}; validation tokens: {len(validation.tokens):,}",
        flush=True,
    )
    model.train()
    for step in range(start + 1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        microbatches = []
        for _ in range(args.gradient_accumulation):
            indices = torch.randint(
                len(training), (args.batch_size,), generator=sampler
            ).tolist()
            rows = [training[index] for index in indices]
            batch = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
            targets = int((batch["labels"][:, 1:] != -100).sum().item())
            microbatches.append((batch, targets))
        total_targets = sum(targets for _, targets in microbatches)
        train_loss = 0.0
        for batch, targets in microbatches:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at step {step}.")
            weight = targets / total_targets
            (loss * weight).backward()
            train_loss += loss.detach().item() * weight
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            metrics = {
                "step": step,
                "train_loss": train_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            print(json.dumps(metrics), flush=True)
            append_json(output / "metrics.jsonl", metrics)
        if step % SAMPLE_EVERY == 0:
            samples = sample_responses(
                model,
                tokenizer,
                prompts,
                device,
                args.sample_max_new_tokens,
                seed=args.seed,
            )
            show_samples(samples, step)
            append_json(output / "samples.jsonl", {"step": step, "samples": samples})
        if step % args.eval_every == 0 or step == args.max_steps:
            metrics = evaluate(
                model, validation, device, args.batch_size, args.eval_batches
            )
            print(f"Validation at step {step}: {json.dumps(metrics)}", flush=True)
            append_json(output / "metrics.jsonl", {"step": step, "validation": metrics})
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                save_checkpoint(
                    output / "best",
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    sampler,
                    step,
                    best_loss,
                    run,
                    device,
                )
        if step % args.save_every == 0:
            save_checkpoint(
                output / f"checkpoint-{step}",
                model,
                tokenizer,
                optimizer,
                scheduler,
                sampler,
                step,
                best_loss,
                run,
                device,
            )
    save_checkpoint(
        output / "model",
        model,
        tokenizer,
        optimizer,
        scheduler,
        sampler,
        step,
        best_loss,
        run,
        device,
    )
    print(
        f"Saved final model to {output / 'model'}; best validation checkpoint to {output / 'best'}.",
        flush=True,
    )


def load_model(path, device):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(
        path, local_files_only=True, attn_implementation="eager"
    )
    return model.to(device), tokenizer


def test_model(args):
    if args.batch_size < 1 or args.max_batches < 0 or args.threads < 1:
        raise ValueError(
            "Batch size and threads must be positive; max-batches must be nonnegative."
        )
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    model, tokenizer = load_model(args.model_dir, device)
    metadata = json.loads((Path(args.data_dir) / "metadata.json").read_text())
    run = json.loads((Path(args.model_dir) / "run.json").read_text())
    if metadata["fingerprint"] != run["dataset_fingerprint"]:
        raise ValueError(
            "The evaluation corpus/tokenizer differs from the training dataset."
        )
    dataset = TokenBlocks(
        args.data_dir, args.split, model.config.n_positions, tokenizer.pad_token_id
    )
    metrics = evaluate(model, dataset, device, args.batch_size, args.max_batches)
    samples = sample_responses(
        model,
        tokenizer,
        read_prompts(args.sample_prompts),
        device,
        args.max_new_tokens,
        seed=args.seed,
    )
    result = {
        "split": args.split,
        **metrics,
        "samples": samples,
        "note": "Perplexity measures language modeling, not factual correctness.",
    }
    destination = (
        Path(args.report)
        if args.report
        else Path(args.model_dir).parent / f"{args.split}-results.json"
    )
    write_json(destination, result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "samples"}, indent=2
        )
    )
    show_samples(samples)
    print(f"Saved evaluation report to {destination}")


def generate(args):
    if args.threads < 1:
        raise ValueError("threads must be positive.")
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    model, tokenizer = load_model(args.model_dir, device)
    samples = sample_responses(
        model,
        tokenizer,
        args.prompt,
        device,
        args.max_new_tokens,
        args.temperature,
        args.seed,
    )
    show_samples(samples)
