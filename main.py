"""CLI entrypoint for the agent-learning pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from config import NumericRange, SandboxConfig
from logging_tools.episode_io import load_episode


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the pipeline."""

    parser = argparse.ArgumentParser(description="Physics sandbox agent-learning pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Collect episodes with a random policy.")
    collect.add_argument("--mode", choices=("1d", "2d"), default="1d")
    collect.add_argument("--episodes", type=int, default=100)
    collect.add_argument("--episode-length", type=int, default=120)
    collect.add_argument("--seed", type=int, default=None)
    collect.add_argument("--dt", type=float, default=None)
    collect.add_argument("--max-force", type=float, default=None)
    collect.add_argument("--mass-min", type=float, default=None)
    collect.add_argument("--mass-max", type=float, default=None)
    collect.add_argument("--output-dir", type=Path, default=Path("data/episodes/train"))

    train = subparsers.add_parser("train", help="Train the 1D acceleration model.")
    train.add_argument("--train-dir", type=Path, required=True)
    train.add_argument("--val-dir", type=Path, required=True)
    train.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--device", type=str, default="cpu")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a trained checkpoint.")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--test-dir", type=Path, required=True)
    evaluate.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    evaluate.add_argument("--max-rollout-steps", type=int, default=50)

    render = subparsers.add_parser("render", help="Replay a logged episode.")
    render.add_argument("--episode", type=Path, required=True)
    render.add_argument("--checkpoint", type=Path, default=None)
    render.add_argument("--width", type=int, default=None)
    render.add_argument("--height", type=int, default=None)
    render.add_argument("--fps", type=int, default=None)
    render.add_argument("--show-mass", action="store_true")
    render.add_argument("--hide-debug", action="store_true")

    return parser.parse_args()


def build_collect_config(args: argparse.Namespace) -> SandboxConfig:
    """Create a collection config from CLI arguments."""

    config = SandboxConfig(mode=args.mode)
    overrides: dict[str, Any] = {
        "mode": args.mode,
        "seed": args.seed,
        "episode_length_limit": args.episode_length,
    }
    if args.dt is not None:
        overrides["dt"] = args.dt
    if args.max_force is not None:
        overrides["max_force"] = args.max_force
    if args.mass_min is not None or args.mass_max is not None:
        minimum = config.mass_range.minimum if args.mass_min is None else args.mass_min
        maximum = config.mass_range.maximum if args.mass_max is None else args.mass_max
        overrides["mass_range"] = NumericRange(minimum, maximum)
    return SandboxConfig(**overrides)


def main() -> None:
    """Dispatch pipeline subcommands."""

    args = parse_args()
    if args.command == "collect":
        if args.mode != "1d":
            raise SystemExit("collect currently supports only `--mode 1d`.")
        from agents.random_policy import RandomUniformPolicy
        from training.collector import TrajectoryCollector

        config = build_collect_config(args)
        policy = RandomUniformPolicy(max_force=config.max_force, seed=config.seed)
        collector = TrajectoryCollector(config=config, policy=policy, output_dir=args.output_dir)
        summary = collector.collect(args.episodes)
        print(f"Collected {summary.episodes_collected} episodes into {summary.output_dir}")
        return

    if args.command == "train":
        try:
            from training.train_dynamics import train_dynamics
        except ImportError as exc:
            raise SystemExit(
                "PyTorch is required for training. Install dependencies with "
                "`python3 -m pip install -r requirements.txt`."
            ) from exc

        result = train_dynamics(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            artifact_dir=args.artifact_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
        )
        print(f"Saved checkpoint to {result['checkpoint_path']}")
        return

    if args.command == "evaluate":
        try:
            from evaluation.evaluate_model import evaluate_model
        except ImportError as exc:
            raise SystemExit(
                "PyTorch is required for evaluation. Install dependencies with "
                "`python3 -m pip install -r requirements.txt`."
            ) from exc

        summary = evaluate_model(
            checkpoint_path=args.checkpoint,
            test_dir=args.test_dir,
            artifact_dir=args.artifact_dir,
            max_rollout_steps=args.max_rollout_steps,
        )
        print(f"Evaluation summary written to {summary['summary_path']}")
        return

    if args.command == "render":
        try:
            from app_controller import ReplayController
        except ImportError as exc:
            raise SystemExit(
                "Pygame is required for replay rendering. Install dependencies with "
                "`python3 -m pip install -r requirements.txt`."
            ) from exc

        episode = load_episode(args.episode)
        config_payload = dict(episode.config)
        if args.width is not None:
            config_payload["screen_width"] = args.width
        if args.height is not None:
            config_payload["screen_height"] = args.height
        if args.fps is not None:
            config_payload["target_fps"] = args.fps
        if args.show_mass:
            config_payload["show_mass"] = True
        if args.hide_debug:
            config_payload["show_debug"] = False
        config_payload["mode"] = episode.mode
        config = SandboxConfig.from_dict(config_payload)
        try:
            controller = ReplayController(episode=episode, config=config, checkpoint_path=args.checkpoint)
        except ImportError as exc:
            raise SystemExit(
                "PyTorch is required when using `render --checkpoint`. Install dependencies with "
                "`python3 -m pip install -r requirements.txt`."
            ) from exc
        controller.run()
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
