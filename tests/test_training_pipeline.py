"""Training and evaluation smoke tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from agents.random_policy import RandomUniformPolicy
from config import SandboxConfig
from training.collector import TrajectoryCollector

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class TrainingPipelineTests(unittest.TestCase):
    """Validate that the training pipeline can run end-to-end."""

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for the training pipeline tests.")
    def test_train_and_evaluate_smoke(self) -> None:
        from evaluation.evaluate_model import evaluate_model
        from training.train_dynamics import train_dynamics

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / "train"
            val_dir = root / "val"
            test_dir = root / "test"
            artifact_dir = root / "artifacts"

            for directory in (train_dir, val_dir, test_dir):
                config = SandboxConfig(mode="1d", seed=4, episode_length_limit=16)
                policy = RandomUniformPolicy(max_force=config.max_force, seed=4)
                collector = TrajectoryCollector(config=config, policy=policy, output_dir=directory)
                collector.collect(episodes=4 if directory == train_dir else 2)

            train_result = train_dynamics(
                train_dir=train_dir,
                val_dir=val_dir,
                artifact_dir=artifact_dir,
                epochs=2,
                batch_size=8,
                learning_rate=1e-3,
            )
            self.assertTrue(train_result["checkpoint_path"].exists())
            self.assertTrue(train_result["metadata_path"].exists())

            eval_summary = evaluate_model(
                checkpoint_path=train_result["checkpoint_path"],
                test_dir=test_dir,
                artifact_dir=artifact_dir,
                max_rollout_steps=10,
            )
            self.assertIn("one_step_acceleration_mae", eval_summary)
            self.assertTrue(Path(eval_summary["summary_path"]).exists())


if __name__ == "__main__":
    unittest.main()
