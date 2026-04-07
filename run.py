from __future__ import annotations

import argparse
import sys
from pathlib import Path

from car_driver.config import AppConfig
from car_driver.training import (
    TrainerSession,
    load_brain_checkpoint,
    save_brain_checkpoint,
    smoke_test,
)
from car_driver.validation import validate_app_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Car Driving PPO trainer and visualizer."
    )
    subparsers = parser.add_subparsers(dest="command")

    gui_parser = subparsers.add_parser("gui", help="Launch the desktop UI.")
    gui_parser.add_argument(
        "--load",
        type=Path,
        default=None,
        help="Optional checkpoint to load on startup.",
    )

    train_parser = subparsers.add_parser(
        "headless-train",
        help="Run PPO training from the command line without the GUI.",
    )
    train_parser.add_argument("--load", type=Path, default=None)
    train_parser.add_argument("--generations", type=int, default=None)
    train_parser.add_argument("--games", type=int, default=None)
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument("--save", type=Path, default=None)

    smoke_parser = subparsers.add_parser(
        "smoke-test",
        help="Run a quick engine and PPO integration smoke test.",
    )
    smoke_parser.add_argument("--seed", type=int, default=11)

    parser.set_defaults(command="gui")
    return parser


def run_headless_training(args: argparse.Namespace) -> int:
    if args.load:
        try:
            startup_payload = load_brain_checkpoint(args.load)
        except Exception as exc:
            print(f"Failed to load checkpoint {args.load}: {exc}")
            return 1
    else:
        startup_payload = None
    config = startup_payload["config"] if startup_payload else AppConfig()
    if startup_payload and startup_payload.get("warnings"):
        print("Checkpoint load warnings:")
        for warning in startup_payload["warnings"]:
            print(f"- {warning}")
    if args.generations is not None:
        config.ppo.target_generations = args.generations
    if args.games is not None:
        config.ppo.games_per_generation = args.games
    if args.seed is not None:
        config.ppo.seed = args.seed
    validation = validate_app_config(config)
    if validation.errors:
        print("Configuration invalid:")
        for error in validation.errors:
            print(f"- {error}")
        return 1
    if validation.warnings:
        print("Configuration warnings:")
        for warning in validation.warnings:
            print(f"- {warning}")

    session = TrainerSession(
        config=config,
        initial_history=startup_payload.get("history") if startup_payload else None,
        initial_state_dict=startup_payload.get("current_state_dict") if startup_payload else None,
        initial_best_state_dict=startup_payload.get("best_state_dict") if startup_payload else None,
        initial_best_metrics=startup_payload.get("best_metrics") if startup_payload else None,
        initial_optimizer_state_dict=startup_payload.get("optimizer_state_dict") if startup_payload else None,
        initial_observation_normalizer_state=(
            startup_payload.get("observation_normalizer_state")
            if startup_payload
            else None
        ),
        initial_best_observation_normalizer_state=(
            startup_payload.get("best_observation_normalizer_state")
            if startup_payload
            else None
        ),
    )
    print(f"Using compute device: {session.device_info.description}")
    for warning in session.device_info.warnings:
        print(f"- {warning}")
    summary = session.train()

    print(
        f"Training finished with status={summary.status}, "
        f"generations={len(summary.history)}, "
        f"best_finish_rate={summary.best_metrics['finish_rate']:.3f}, "
        f"best_mean_score={summary.best_metrics['mean_score']:.3f}"
    )

    if args.save:
        save_brain_checkpoint(
            path=args.save,
            config=config,
            state_dict=summary.best_state_dict,
            best_metrics=summary.best_metrics,
            history=summary.history,
            source_label="best",
            current_state_dict=summary.current_state_dict,
            best_state_dict=summary.best_state_dict,
            optimizer_state_dict=summary.current_optimizer_state_dict,
            observation_normalizer_state=summary.current_normalizer_state,
            best_observation_normalizer_state=summary.best_normalizer_state,
            metadata=summary.checkpoint_metadata,
        )
        print(f"Saved training checkpoint to {args.save}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "smoke-test":
        ok, message = smoke_test(seed=args.seed)
        print(message)
        return 0 if ok else 1

    if args.command == "headless-train":
        return run_headless_training(args)

    startup_payload = None
    if getattr(args, "load", None):
        try:
            startup_payload = load_brain_checkpoint(args.load)
        except Exception as exc:
            print(f"Failed to load checkpoint {args.load}: {exc}")
            return 1
    from car_driver.tk_gui import launch_gui

    return launch_gui(startup_payload=startup_payload)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
