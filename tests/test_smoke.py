from __future__ import annotations

import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from car_driver.config import AppConfig, LayerConfig
from car_driver.environment import (
    OBSERVATION_NAMES,
    CarDriverEnv,
    build_observation_names,
    track_self_intersects,
)
from car_driver.training import (
    TorchDeviceInfo,
    TrainerSession,
    load_brain_checkpoint,
    resolve_torch_device,
    save_brain_checkpoint,
)
from car_driver.ui_app import MainApplication, describe_evaluation_outcome
from car_driver.ui_controls import (
    adaptive_grid_columns_for_width,
    compact_panel_mode_for_width,
)
from car_driver.validation import validate_app_config


def fast_test_config() -> AppConfig:
    config = AppConfig()
    config.track.target_course_length = 120.0
    config.track.min_straight_length = 140.0
    config.track.max_straight_length = 140.0
    config.physics.max_steps = 120
    config.ppo.target_generations = 1
    config.ppo.games_per_generation = 3
    config.ppo.ppo_epochs = 2
    config.ppo.minibatch_size = 32
    return config


class CarDriverSmokeTests(unittest.TestCase):
    def test_resolve_torch_device_falls_back_to_cpu_after_probe_error(self) -> None:
        def fake_probe(device_type: str, device_index: int | None = None):
            if device_type == "cuda":
                raise RuntimeError("driver init failed")
            return None

        with mock.patch("car_driver.training._probe_gpu_backend", side_effect=fake_probe):
            device_info = resolve_torch_device()

        self.assertEqual(device_info.backend, "cpu")
        self.assertEqual(device_info.device.type, "cpu")
        self.assertTrue(any("CUDA probe failed" in warning for warning in device_info.warnings))

    def test_resolve_torch_device_returns_requested_gpu_when_probe_succeeds(self) -> None:
        expected = TorchDeviceInfo(
            device=torch.device("cuda"),
            backend="cuda",
            description="CUDA GPU (Mock)",
        )
        with mock.patch("car_driver.training._probe_gpu_backend", return_value=expected):
            device_info = resolve_torch_device("cuda")

        self.assertEqual(device_info.backend, "cuda")
        self.assertEqual(str(device_info.device), "cuda")

    def test_parallel_env_setting_is_configurable_and_capped_by_generation_size(self) -> None:
        config = fast_test_config()
        config.ppo.parallel_envs = 2
        session = TrainerSession(config=config, device="cpu")
        self.assertEqual(session.parallel_env_count, 2)

        config.ppo.parallel_envs = 10
        session = TrainerSession(config=config, device="cpu")
        self.assertEqual(session.parallel_env_count, config.ppo.games_per_generation)

    def test_environment_observation_shape(self) -> None:
        env = CarDriverEnv(seed=4)
        obs = env.reset()
        self.assertEqual(obs.shape[0], len(OBSERVATION_NAMES))
        next_obs, reward, done, info = env.step(np.array([0.4, 0.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(next_obs.shape[0], len(OBSERVATION_NAMES))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn("event", info)

    def test_lidar_values_stay_normalized_and_react_to_edge_proximity(self) -> None:
        config = fast_test_config()
        config.track.road_width = 10.0
        env = CarDriverEnv(config.physics, config.rewards, config.sensors, config.track, seed=3)
        env.reset(seed=3)
        assert env.state is not None
        env.state.x = 4.0
        env.state.y = 8.0
        env.state.heading = 0.0
        env.state.nearest_index = 0
        env._update_track_state()

        lidar = env.state.lidar_values
        self.assertTrue(np.all(lidar >= 0.0))
        self.assertTrue(np.all(lidar <= 1.0))
        self.assertLess(lidar[-1], lidar[0])

    def test_track_generation_has_expected_length_and_no_self_intersection(self) -> None:
        config = fast_test_config()
        config.track.target_course_length = 450.0
        config.track.min_straight_length = 40.0
        config.track.max_straight_length = 60.0
        env = CarDriverEnv(config.physics, config.rewards, config.sensors, config.track, seed=5)
        env.reset(seed=5)
        snapshot = env.snapshot()
        track = snapshot["track"]
        self.assertIsNotNone(track)
        self.assertGreaterEqual(float(track["target_length"]), 430.0)
        self.assertAlmostEqual(float(track["road_width"]), config.track.road_width, places=5)
        self.assertFalse(track_self_intersects(track["centerline"], float(track["point_spacing"])))

    def test_stalled_episode_terminates(self) -> None:
        config = fast_test_config()
        config.physics.stalled_timeout_seconds = 0.3
        config.physics.initial_fuel_liters = 0.0
        env = CarDriverEnv(config.physics, config.rewards, config.sensors, config.track, seed=7)
        env.reset(seed=7)
        done = False
        info = {}
        while not done:
            _, _, done, info = env.step(np.zeros(4, dtype=np.float32))
        self.assertTrue(info.get("stalled"))
        self.assertEqual(info.get("event"), "stall")

    def test_fuel_depletion_disables_gas_acceleration(self) -> None:
        config = fast_test_config()
        config.physics.initial_fuel_liters = 0.05
        config.physics.fuel_tank_liters = 0.05
        config.physics.fuel_usage_lps = 2.0
        env = CarDriverEnv(config.physics, config.rewards, config.sensors, config.track, seed=9)
        env.reset(seed=9)
        for _ in range(3):
            env.step(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        assert env.state is not None
        env.state.fuel = 0.0
        speed_before = env.state.speed
        env.step(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.assertLessEqual(env.state.speed, speed_before + 1e-6)

    def test_trainer_runs_and_checkpoint_roundtrip(self) -> None:
        config = fast_test_config()
        summary = TrainerSession(config=config, device="cpu").train()
        self.assertEqual(summary.status, "completed")
        self.assertEqual(len(summary.history), 1)
        self.assertFalse(summary.history[0].partial_generation)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "brain.pt"
            save_brain_checkpoint(
                path=checkpoint_path,
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
            payload = load_brain_checkpoint(checkpoint_path)
            self.assertEqual(len(payload["history"]), 1)
            self.assertIn("state_dict", payload)
            self.assertIn("current_state_dict", payload)
            self.assertIn("best_state_dict", payload)
            self.assertIn("optimizer_state_dict", payload)
            self.assertIn("metadata", payload)
            self.assertEqual(payload["metadata"]["schema_version"], 3)
            self.assertEqual(
                payload["observation_names"],
                build_observation_names(config.sensors),
            )

            resumed = TrainerSession(
                config=payload["config"],
                device="cpu",
                initial_history=payload["history"],
                initial_state_dict=payload["current_state_dict"],
                initial_best_state_dict=payload["best_state_dict"],
                initial_best_metrics=payload["best_metrics"],
                initial_optimizer_state_dict=payload["optimizer_state_dict"],
                initial_observation_normalizer_state=payload["observation_normalizer_state"],
                initial_best_observation_normalizer_state=payload["best_observation_normalizer_state"],
            ).train(generations=1)
            self.assertEqual(resumed.status, "completed")
            self.assertEqual(len(resumed.history), 2)
            self.assertEqual(resumed.history[-1].generation_index, 1)

    def test_load_checkpoint_reconciles_network_config_to_saved_weights(self) -> None:
        config = fast_test_config()
        summary = TrainerSession(config=config, device="cpu").train()

        mismatched_config = AppConfig.from_dict(config.to_dict())
        mismatched_config.network.hidden_layers = [
            LayerConfig(13, "tanh"),
            LayerConfig(60, "relu"),
            LayerConfig(11, "relu"),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "legacy_bad_config.pt"
            torch.save(
                {
                    "config": mismatched_config.to_dict(),
                    "state_dict": summary.best_state_dict,
                    "current_state_dict": summary.current_state_dict,
                    "best_state_dict": summary.best_state_dict,
                    "optimizer_state_dict": summary.current_optimizer_state_dict,
                    "best_metrics": summary.best_metrics,
                    "history": [report.to_dict() for report in summary.history],
                    "observation_normalizer_state": summary.current_normalizer_state,
                    "best_observation_normalizer_state": summary.best_normalizer_state,
                },
                checkpoint_path,
            )

            payload = load_brain_checkpoint(checkpoint_path)
            self.assertEqual(
                [layer.units for layer in payload["config"].network.hidden_layers],
                [12, 60, 10],
            )
            self.assertTrue(payload["warnings"])

            resumed = TrainerSession(
                config=payload["config"],
                device="cpu",
                initial_history=payload["history"],
                initial_state_dict=payload["current_state_dict"],
                initial_best_state_dict=payload["best_state_dict"],
                initial_best_metrics=payload["best_metrics"],
                initial_optimizer_state_dict=payload["optimizer_state_dict"],
                initial_observation_normalizer_state=payload["observation_normalizer_state"],
                initial_best_observation_normalizer_state=payload["best_observation_normalizer_state"],
            )
            self.assertEqual(resumed.model.observation_dim, len(build_observation_names(config.sensors)))

    def test_save_checkpoint_reconciles_config_before_writing(self) -> None:
        config = fast_test_config()
        summary = TrainerSession(config=config, device="cpu").train()

        mismatched_config = AppConfig.from_dict(config.to_dict())
        mismatched_config.network.hidden_layers = [
            LayerConfig(13, "tanh"),
            LayerConfig(60, "relu"),
            LayerConfig(11, "relu"),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "saved.pt"
            save_brain_checkpoint(
                path=checkpoint_path,
                config=mismatched_config,
                state_dict=summary.best_state_dict,
                best_metrics=summary.best_metrics,
                history=summary.history,
                source_label="best",
                current_state_dict=summary.current_state_dict,
                best_state_dict=summary.best_state_dict,
                optimizer_state_dict=summary.current_optimizer_state_dict,
                observation_normalizer_state=summary.current_normalizer_state,
                best_observation_normalizer_state=summary.best_normalizer_state,
            )
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(
                [
                    int(layer["units"])
                    for layer in raw["config"]["network"]["hidden_layers"]
                ],
                [12, 60, 10],
            )

    def test_validation_catches_cross_field_errors(self) -> None:
        config = AppConfig()
        config.sensors.lidar_beam_count = 12
        config.track.road_width = config.physics.car_width - 0.1
        config.track.min_turn_radius = 50.0
        config.track.max_turn_radius = 20.0
        validation = validate_app_config(config)
        self.assertFalse(validation.is_valid)
        self.assertTrue(any("odd" in error for error in validation.errors))
        self.assertTrue(any("Road width" in error for error in validation.errors))
        self.assertTrue(any("turn radius" in error.lower() for error in validation.errors))

    def test_adaptive_control_panel_columns_switch_by_width(self) -> None:
        self.assertEqual(adaptive_grid_columns_for_width(260), 1)
        self.assertEqual(adaptive_grid_columns_for_width(320), 2)
        self.assertEqual(adaptive_grid_columns_for_width(520), 2)

    def test_compact_panel_mode_switches_by_width(self) -> None:
        self.assertTrue(compact_panel_mode_for_width(260))
        self.assertFalse(compact_panel_mode_for_width(320))
        self.assertFalse(compact_panel_mode_for_width(520))

    def test_poll_training_queue_survives_bridge_reset(self) -> None:
        app = object.__new__(MainApplication)
        app.closed = False
        scheduled: list[tuple[int, object]] = []
        app.root = SimpleNamespace(after=lambda delay, callback: scheduled.append((delay, callback)))
        app.training_bridge = SimpleNamespace(queue=queue.Queue())
        app.training_bridge.queue.put(("finished", "paused"))

        handled: list[str] = []
        app._handle_generation_report = lambda payload: handled.append("generation")

        def mark_finished(payload: str) -> None:
            handled.append(payload)
            app.training_bridge = None

        app._training_finished = mark_finished
        app._training_failed = lambda payload: handled.append(f"failed:{payload}")

        MainApplication._poll_training_queue(app)

        self.assertEqual(handled, ["paused"])
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0][0], 100)

    def test_evaluation_outcome_monitor_tracks_success_and_failure(self) -> None:
        finished = describe_evaluation_outcome(
            {"event": "finished", "finished": True, "score": 42.5, "distance_travelled": 3000.0, "elapsed_time": 78.2}
        )
        self.assertEqual(finished["kind"], "success")
        self.assertEqual(finished["counter_key"], "finished")
        self.assertIn("FINISH", finished["headline"])

        crashed = describe_evaluation_outcome(
            {"event": "crash", "crashed": True, "score": -12.0, "distance_travelled": 180.0, "speed": 4.6}
        )
        self.assertEqual(crashed["kind"], "failure")
        self.assertEqual(crashed["counter_key"], "crashed")
        self.assertIn("CRASH", crashed["headline"])


if __name__ == "__main__":
    unittest.main()
