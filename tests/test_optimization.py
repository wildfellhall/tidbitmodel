"""Learning-rate endpoints, GPU selection, and loss-scaled optimizer updates."""

import math
import unittest
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import patch

import torch

from main import parser
from one_piece import CONFIG
from tidbit.training import (
    configure_training,
    learning_rate_factor,
    training_configuration,
    update_parameters,
)


class OptimizationTests(unittest.TestCase):
    def metadata(self, passages=824207):
        return {
            "format_version": 2,
            "context": 256,
            "passage_counts": {"train": passages},
        }

    def test_default_budget_warmup_and_batch_match_the_audited_corpus(self):
        args = parser().parse_args(["train", "--device", "cpu"])
        configure_training(args, self.metadata())
        self.assertEqual(args.batch_size * args.gradient_accumulation, 32)
        self.assertEqual(args.max_steps, 3 * math.ceil(824207 / 32))
        self.assertEqual(args.warmup_steps, 1546)
        self.assertEqual(args.learning_rate, 6e-4)
        self.assertEqual(args.precision, "float32")
        self.assertFalse(args.fused_optimizer)

    def test_notebook_and_cli_share_the_same_training_defaults(self):
        standalone = SimpleNamespace(**CONFIG)
        standalone.device = "cpu"
        modular = parser().parse_args(["train", "--device", "cpu"])
        for args in (standalone, modular):
            configure_training(args, self.metadata())
        self.assertEqual(
            training_configuration(standalone), training_configuration(modular)
        )

    def test_zero_step_budget_recalculates_for_different_corpus_and_epochs(self):
        args = parser().parse_args(
            ["train", "--device", "cpu", "--max-steps", "0", "--epochs", "2"]
        )
        configure_training(args, self.metadata(100))
        self.assertEqual(args.steps_per_epoch, 4)
        self.assertEqual(args.max_steps, 8)
        self.assertEqual(args.warmup_steps, 1)

    def test_cosine_reaches_peak_and_floor_on_actual_optimizer_updates(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=6e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: learning_rate_factor(step, 3, 11, 0.1)
        )
        used = []
        for _ in range(11):
            used.append(optimizer.param_groups[0]["lr"])
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(used[0], 2e-4)
        self.assertAlmostEqual(used[2], 6e-4)
        self.assertAlmostEqual(used[-1], 6e-5)
        self.assertTrue(all(a >= b for a, b in pairwise(used[2:])))
        self.assertEqual(learning_rate_factor(0, 0, 11, 0.1), 1)
        self.assertAlmostEqual(learning_rate_factor(10, 0, 11, 0.1), 0.1)
        self.assertEqual(learning_rate_factor(0, 0, 1, 0.1), 1)

    def test_cuda_auto_precision_requires_native_bf16_support(self):
        for native, expected in ((True, "bfloat16"), (False, "float16")):
            args = parser().parse_args(["train"])
            with (
                patch(
                    "tidbit.training.choose_device", return_value=torch.device("cuda")
                ),
                patch("torch.cuda.is_bf16_supported", return_value=native) as supported,
            ):
                configure_training(args, self.metadata())
            supported.assert_called_once_with(including_emulation=False)
            self.assertEqual(args.precision, expected)
            self.assertEqual(args.attention_implementation, "sdpa")
            self.assertTrue(args.fused_optimizer)

    def test_invalid_schedule_values_are_rejected_before_training(self):
        for arguments in (
            ("--warmup-ratio", "1.0"),
            ("--min-lr-ratio", "-0.1"),
            ("--warmup-steps", "-1"),
            ("--precision", "float16"),
        ):
            args = parser().parse_args(["train", "--device", "cpu", *arguments])
            with self.assertRaises(ValueError):
                configure_training(args, self.metadata())

    def test_loss_scaler_skips_overflow_then_unscales_before_clipping(self):
        # CPU GradScaler exercises the same public API and overflow state machine
        # without pretending this machine has a CUDA GPU.
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=16.0)
        scaler.scale(model.weight.sum() * float("inf")).backward()
        _, updated = update_parameters(model, optimizer, scaler)
        self.assertFalse(updated)
        self.assertEqual(model.weight.item(), 1.0)
        self.assertEqual(scaler.get_scale(), 8.0)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(model.weight.sum() * 2.0).backward()
        norm, updated = update_parameters(model, optimizer, scaler)
        self.assertTrue(updated)
        self.assertAlmostEqual(norm.item(), 2.0)
        self.assertAlmostEqual(model.weight.item(), 0.9, places=6)


if __name__ == "__main__":
    unittest.main()
