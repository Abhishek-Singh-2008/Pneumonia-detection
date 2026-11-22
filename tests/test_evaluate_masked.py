import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.config import METRICS_COLUMNS
from src.evaluate_masked import _validate_checkpoint_metadata, build_comparison_table


class BuildComparisonTableTests(unittest.TestCase):
    def test_builds_final_matched_test_comparison_with_attention(self):
        stage1 = pd.DataFrame([{
            "model": "custom_cnn", "data_setting": "unmasked", "split": "test",
            "accuracy": 0.80, "precision": 0.76, "recall": 0.98, "f1": 0.86,
            "auc": 0.90, "tn": 117, "fp": 117, "fn": 7, "tp": 383,
            "attention_in_lung": None,
        }])
        stage3 = pd.DataFrame([{
            "model": "densenet121", "data_setting": "unmasked", "split": "test",
            "accuracy": 0.87, "precision": 0.83, "recall": 1.00, "f1": 0.91,
            "auc": 0.97, "tn": 156, "fp": 78, "fn": 1, "tp": 389,
            "attention_in_lung": 0.40,
        }])
        stage4 = pd.DataFrame([{
            "model": "densenet121", "data_setting": "masked", "split": "test",
            "accuracy": 0.90, "precision": 0.88, "recall": 0.95, "f1": 0.91,
            "auc": 0.96, "tn": 180, "fp": 54, "fn": 20, "tp": 370,
            "attention_in_lung": 0.60,
        }])

        result = build_comparison_table(stage1, stage3, stage4)

        self.assertEqual(list(result.columns), list(METRICS_COLUMNS))
        self.assertEqual(
            list(zip(result["model"], result["data_setting"])),
            [("custom_cnn", "unmasked"), ("densenet121", "unmasked"), ("densenet121", "masked")],
        )
        self.assertTrue(pd.isna(result.iloc[0]["attention_in_lung"]))
        self.assertEqual(result["attention_in_lung"].iloc[1:].tolist(), [0.40, 0.60])

    def test_rejects_missing_attention_from_stage3(self):
        stage1 = pd.DataFrame([{
            "model": "custom_cnn", "data_setting": "unmasked", "split": "test",
            "attention_in_lung": None,
        }])
        stage3 = pd.DataFrame([{
            "model": "densenet121", "data_setting": "unmasked", "split": "test",
            "attention_in_lung": None,
        }])
        stage4 = pd.DataFrame([{
            "model": "densenet121", "data_setting": "masked", "split": "test",
            "attention_in_lung": 0.6,
        }])

        with self.assertRaisesRegex(ValueError, "missing its attention_in_lung"):
            build_comparison_table(stage1, stage3, stage4)


class CheckpointMetadataTests(unittest.TestCase):
    def test_requires_selected_unmasked_run(self):
        checkpoint = {
            "extra": {
                "model": "densenet121",
                "phase": "B (all layers fine-tuned)",
                "run": "lr1e-4_bs16",
                "batch_size": 16,
            }
        }
        with patch("src.evaluate_masked.torch.load", return_value=checkpoint):
            _validate_checkpoint_metadata(Path("unmasked.pth"), "unmasked")

    def test_rejects_non_masked_phase_b_checkpoint(self):
        checkpoint = {
            "extra": {
                "model": "densenet121",
                "phase": "B (all layers fine-tuned)",
                "data_setting": "unmasked",
            }
        }
        with patch("src.evaluate_masked.torch.load", return_value=checkpoint):
            with self.assertRaisesRegex(ValueError, "does not match the selected masked"):
                _validate_checkpoint_metadata(Path("wrong.pth"), "masked")


if __name__ == "__main__":
    unittest.main()
