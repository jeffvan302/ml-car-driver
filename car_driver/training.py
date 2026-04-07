from __future__ import annotations

import copy
import math
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from car_driver.config import AppConfig, LayerConfig, NetworkConfig
from car_driver.environment import CarDriverEnv, build_observation_names
from car_driver.ppo import ACTION_NAMES, ActorCritic


@dataclass
class GenerationReport:
    generation_index: int
    episodes_run: int
    finish_rate: float
    best_score: float
    mean_score: float
    mean_distance: float
    scores: list[float] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    finished_episodes: int = 0
    total_steps: int = 0
    current_state_dict: dict[str, torch.Tensor] = field(default_factory=dict)
    best_state_dict: dict[str, torch.Tensor] = field(default_factory=dict)
    best_updated: bool = False
    partial_generation: bool = False
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    current_normalizer_state: dict[str, Any] = field(default_factory=dict)
    best_normalizer_state: dict[str, Any] = field(default_factory=dict)
    course_length: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_index": self.generation_index,
            "episodes_run": self.episodes_run,
            "finish_rate": self.finish_rate,
            "best_score": self.best_score,
            "mean_score": self.mean_score,
            "mean_distance": self.mean_distance,
            "scores": list(self.scores),
            "distances": list(self.distances),
            "finished_episodes": self.finished_episodes,
            "total_steps": self.total_steps,
            "best_updated": self.best_updated,
            "partial_generation": self.partial_generation,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "current_normalizer_state": clone_normalizer_state(
                self.current_normalizer_state
            ),
            "best_normalizer_state": clone_normalizer_state(
                self.best_normalizer_state
            ),
            "course_length": self.course_length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationReport":
        return cls(
            generation_index=int(data.get("generation_index", 0)),
            episodes_run=int(data.get("episodes_run", 0)),
            finish_rate=float(
                data.get("finish_rate", data.get("landing_rate", 0.0))
            ),
            best_score=float(data.get("best_score", 0.0)),
            mean_score=float(data.get("mean_score", 0.0)),
            mean_distance=float(data.get("mean_distance", 0.0)),
            scores=list(data.get("scores", [])),
            distances=list(data.get("distances", [])),
            finished_episodes=int(
                data.get("finished_episodes", data.get("landed_episodes", 0))
            ),
            total_steps=int(data.get("total_steps", 0)),
            best_updated=bool(data.get("best_updated", False)),
            partial_generation=bool(data.get("partial_generation", False)),
            policy_loss=float(data.get("policy_loss", 0.0)),
            value_loss=float(data.get("value_loss", 0.0)),
            entropy=float(data.get("entropy", 0.0)),
            current_normalizer_state=clone_normalizer_state(
                data.get("current_normalizer_state", {})
            ),
            best_normalizer_state=clone_normalizer_state(
                data.get("best_normalizer_state", {})
            ),
            course_length=float(data.get("course_length", 0.0)),
        )


@dataclass
class TrainingSummary:
    status: str
    history: list[GenerationReport]
    current_state_dict: dict[str, torch.Tensor]
    best_state_dict: dict[str, torch.Tensor]
    best_metrics: dict[str, float]
    current_optimizer_state_dict: dict[str, Any]
    current_normalizer_state: dict[str, Any]
    best_normalizer_state: dict[str, Any]
    checkpoint_metadata: dict[str, Any]


@dataclass
class EpisodeTrajectory:
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    score: float = 0.0
    final_info: dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        log_prob: float,
        value: float,
    ) -> None:
        self.observations.append(np.asarray(observation, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.score += float(reward)


@dataclass
class ActiveEpisode:
    env: CarDriverEnv
    raw_observation: np.ndarray
    trajectory: EpisodeTrajectory = field(default_factory=EpisodeTrajectory)


@dataclass
class TorchDeviceInfo:
    device: torch.device
    backend: str
    description: str
    warnings: list[str] = field(default_factory=list)

    @property
    def is_gpu(self) -> bool:
        return self.backend != "cpu"


class ObservationNormalizer:
    def __init__(
        self,
        dimension: int,
        clip_value: float = 5.0,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.dimension = dimension
        self.clip_value = float(clip_value)
        self.mean = np.zeros(dimension, dtype=np.float64)
        self.var = np.ones(dimension, dtype=np.float64)
        self.count = 1e-4

        if state:
            self.mean = np.asarray(state.get("mean", self.mean), dtype=np.float64)
            self.var = np.maximum(
                np.asarray(state.get("var", self.var), dtype=np.float64),
                1e-6,
            )
            self.count = float(state.get("count", self.count))
            self.clip_value = float(state.get("clip_value", self.clip_value))

    def update(self, batch: np.ndarray) -> None:
        array = np.asarray(batch, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[0] == 0:
            return

        batch_mean = array.mean(axis=0)
        batch_var = array.var(axis=0)
        batch_count = array.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        mean_a = self.var * self.count
        mean_b = batch_var * batch_count
        m2 = mean_a + mean_b + delta**2 * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = np.maximum(m2 / total_count, 1e-6)
        self.count = total_count

    def normalize(self, batch: np.ndarray) -> np.ndarray:
        array = np.asarray(batch, dtype=np.float32)
        normalized = (array - self.mean.astype(np.float32)) / np.sqrt(
            self.var.astype(np.float32) + 1e-8
        )
        return np.clip(normalized, -self.clip_value, self.clip_value).astype(
            np.float32
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.astype(np.float32).copy(),
            "var": self.var.astype(np.float32).copy(),
            "count": float(self.count),
            "clip_value": float(self.clip_value),
        }


def _cpu_device_info(warnings: list[str] | None = None) -> TorchDeviceInfo:
    return TorchDeviceInfo(
        device=torch.device("cpu"),
        backend="cpu",
        description="CPU",
        warnings=list(warnings or []),
    )


def _probe_gpu_backend(
    device_type: str,
    device_index: int | None = None,
) -> TorchDeviceInfo | None:
    backend_module = getattr(torch, device_type, None)
    if backend_module is None:
        return None

    is_available = getattr(backend_module, "is_available", None)
    if not callable(is_available):
        return None

    available = bool(is_available())
    if not available:
        return None

    device_name = device_type if device_index is None else f"{device_type}:{device_index}"
    device = torch.device(device_name)
    probe = torch.zeros(4, device=device, dtype=torch.float32)
    probe = probe + 1.0
    _ = float(probe.sum().item())

    synchronize = getattr(backend_module, "synchronize", None)
    if callable(synchronize):
        synchronize()

    backend_label = "rocm" if device_type == "cuda" and getattr(torch.version, "hip", None) else device_type
    readable_backend = {
        "cuda": "CUDA GPU",
        "rocm": "ROCm GPU",
        "xpu": "XPU GPU",
    }.get(backend_label, f"{backend_label.upper()} device")

    accelerator_name = None
    get_device_name = getattr(backend_module, "get_device_name", None)
    if callable(get_device_name):
        try:
            accelerator_name = str(get_device_name(device_index or 0))
        except Exception:
            accelerator_name = None

    description = readable_backend
    if accelerator_name:
        description = f"{description} ({accelerator_name})"

    return TorchDeviceInfo(
        device=device,
        backend=backend_label,
        description=description,
    )


def resolve_torch_device(
    requested_device: str | torch.device | None = None,
) -> TorchDeviceInfo:
    warnings: list[str] = []

    if requested_device is not None:
        try:
            requested = torch.device(requested_device)
        except Exception as exc:
            return _cpu_device_info(
                [f"Requested compute device {requested_device!r} was invalid: {exc}"]
            )
        if requested.type == "cpu":
            return _cpu_device_info()
        if requested.type not in {"cuda", "xpu"}:
            return TorchDeviceInfo(
                device=requested,
                backend=requested.type,
                description=f"{requested.type.upper()} device ({requested})",
            )
        try:
            device_info = _probe_gpu_backend(requested.type, requested.index)
            if device_info is not None:
                return device_info
            return _cpu_device_info(
                [f"Requested {requested.type.upper()} device is not available. Falling back to CPU."]
            )
        except Exception as exc:
            return _cpu_device_info(
                [f"{requested.type.upper()} probe failed: {exc}. Falling back to CPU."]
            )

    for candidate in ("cuda", "xpu"):
        try:
            device_info = _probe_gpu_backend(candidate)
            if device_info is not None:
                return device_info
        except Exception as exc:
            warnings.append(
                f"{candidate.upper()} probe failed: {exc}. Falling back to the next device."
            )

    return _cpu_device_info(warnings)


def seed_torch_backends(seed: int) -> None:
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    xpu_backend = getattr(torch, "xpu", None)
    xpu_manual_seed_all = getattr(xpu_backend, "manual_seed_all", None)
    xpu_is_available = getattr(xpu_backend, "is_available", None)
    if callable(xpu_manual_seed_all) and callable(xpu_is_available):
        try:
            if xpu_is_available():
                xpu_manual_seed_all(seed)
        except Exception:
            pass


class RolloutBuffer:
    def __init__(self) -> None:
        self.episodes: list[EpisodeTrajectory] = []

    def add_episode(self, trajectory: EpisodeTrajectory) -> None:
        if trajectory.rewards:
            self.episodes.append(trajectory)

    def __len__(self) -> int:
        return sum(len(episode.rewards) for episode in self.episodes)

    def as_tensors(
        self,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        log_probs: list[np.ndarray] = []
        advantages: list[np.ndarray] = []
        returns: list[np.ndarray] = []

        for episode in self.episodes:
            rewards = np.asarray(episode.rewards, dtype=np.float32)
            values = np.asarray(episode.values, dtype=np.float32)
            episode_advantages = np.zeros_like(rewards, dtype=np.float32)

            last_gae = 0.0
            next_value = 0.0
            for step in reversed(range(len(rewards))):
                delta = rewards[step] + gamma * next_value - values[step]
                last_gae = delta + gamma * gae_lambda * last_gae
                episode_advantages[step] = last_gae
                next_value = values[step]

            observations.append(np.asarray(episode.observations, dtype=np.float32))
            actions.append(np.asarray(episode.actions, dtype=np.float32))
            log_probs.append(np.asarray(episode.log_probs, dtype=np.float32))
            advantages.append(episode_advantages)
            returns.append(episode_advantages + values)

        return {
            "observations": torch.as_tensor(
                np.concatenate(observations, axis=0),
                dtype=torch.float32,
                device=device,
            ),
            "actions": torch.as_tensor(
                np.concatenate(actions, axis=0),
                dtype=torch.float32,
                device=device,
            ),
            "log_probs": torch.as_tensor(
                np.concatenate(log_probs, axis=0),
                dtype=torch.float32,
                device=device,
            ),
            "advantages": torch.as_tensor(
                np.concatenate(advantages, axis=0),
                dtype=torch.float32,
                device=device,
            ),
            "returns": torch.as_tensor(
                np.concatenate(returns, axis=0),
                dtype=torch.float32,
                device=device,
            ),
        }


def clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def clone_optimizer_state_dict(state_dict: dict[str, Any] | None) -> dict[str, Any]:
    if not state_dict:
        return {}
    return clone_to_cpu(state_dict)


def clone_normalizer_state(state: dict[str, Any] | None) -> dict[str, Any]:
    state = state or {}
    mean = state.get("mean")
    var = state.get("var")
    if mean is None or var is None:
        return {}
    return {
        "mean": np.asarray(mean, dtype=np.float32).copy(),
        "var": np.asarray(var, dtype=np.float32).copy(),
        "count": float(state.get("count", 1e-4)),
        "clip_value": float(state.get("clip_value", 5.0)),
    }


def normalize_observation_array(
    observation: np.ndarray,
    normalizer_state: dict[str, Any] | None,
) -> np.ndarray:
    if not normalizer_state:
        return np.asarray(observation, dtype=np.float32)
    observation_array = np.asarray(observation, dtype=np.float32)
    try:
        normalizer = ObservationNormalizer(
            dimension=observation_array.shape[-1],
            state=normalizer_state,
        )
        return np.asarray(normalizer.normalize(observation_array), dtype=np.float32)
    except Exception:
        return observation_array


def metric_value(best_metrics: dict[str, float], key: str) -> float:
    if key in best_metrics:
        return float(best_metrics[key])
    if key == "finish_rate" and "landing_rate" in best_metrics:
        return float(best_metrics["landing_rate"])
    return float("-inf") if key == "mean_score" else -1.0


def is_better_candidate(candidate: dict[str, float], best: dict[str, float]) -> bool:
    candidate_finish = metric_value(candidate, "finish_rate")
    best_finish = metric_value(best, "finish_rate")
    candidate_score = metric_value(candidate, "mean_score")
    best_score = metric_value(best, "mean_score")
    if candidate_finish > best_finish + 1e-9:
        return True
    if math.isclose(candidate_finish, best_finish, rel_tol=1e-9, abs_tol=1e-9):
        return candidate_score > best_score
    return False


def _network_units(network_config: NetworkConfig) -> list[int]:
    return [int(layer.units) for layer in network_config.hidden_layers]


def _extract_linear_shapes(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> list[tuple[int, int]]:
    layer_shapes: list[tuple[int, tuple[int, int]]] = []
    for key, value in state_dict.items():
        if not key.startswith(f"{prefix}.") or not key.endswith(".weight"):
            continue
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            continue
        parts = key.split(".")
        if len(parts) < 3:
            continue
        try:
            module_index = int(parts[1])
        except ValueError:
            continue
        shape = (int(value.shape[0]), int(value.shape[1]))
        layer_shapes.append((module_index, shape))
    return [shape for _, shape in sorted(layer_shapes, key=lambda item: item[0])]


def infer_network_signature_from_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    actor_shapes = _extract_linear_shapes(state_dict, "actor")
    critic_shapes = _extract_linear_shapes(state_dict, "critic")
    if not actor_shapes:
        raise ValueError("Checkpoint is missing actor weight tensors.")

    for previous, current in zip(actor_shapes, actor_shapes[1:]):
        if previous[0] != current[1]:
            raise ValueError("Checkpoint actor layers have inconsistent tensor shapes.")

    input_dim = actor_shapes[0][1]
    hidden_units = [shape[0] for shape in actor_shapes[:-1]]
    output_dim = actor_shapes[-1][0]

    if critic_shapes:
        for previous, current in zip(critic_shapes, critic_shapes[1:]):
            if previous[0] != current[1]:
                raise ValueError(
                    "Checkpoint critic layers have inconsistent tensor shapes."
                )
        critic_input_dim = critic_shapes[0][1]
        critic_hidden_units = [shape[0] for shape in critic_shapes[:-1]]
        critic_output_dim = critic_shapes[-1][0]
        if critic_input_dim != input_dim or critic_hidden_units != hidden_units:
            raise ValueError(
                "Checkpoint actor and critic network shapes do not match."
            )
        if critic_output_dim != 1:
            raise ValueError(
                "Checkpoint critic output size is incompatible with this project."
            )

    log_std = state_dict.get("log_std")
    if isinstance(log_std, torch.Tensor) and int(log_std.numel()) != output_dim:
        raise ValueError(
            "Checkpoint action standard deviation shape does not match actor output."
        )

    return {
        "input_dim": input_dim,
        "hidden_units": hidden_units,
        "output_dim": output_dim,
    }


def infer_network_config_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    fallback_network: NetworkConfig | None = None,
) -> NetworkConfig:
    signature = infer_network_signature_from_state_dict(state_dict)
    fallback_layers = (
        list(fallback_network.hidden_layers)
        if fallback_network is not None
        else []
    )
    hidden_layers = [
        LayerConfig(
            units=units,
            activation=(
                fallback_layers[index].activation
                if index < len(fallback_layers)
                else "relu"
            ),
        )
        for index, units in enumerate(signature["hidden_units"])
    ]
    return NetworkConfig(
        hidden_layers=hidden_layers,
        output_activation=(
            fallback_network.output_activation
            if fallback_network is not None
            else "tanh"
        ),
    )


def _signature_text(signature: dict[str, Any]) -> str:
    return (
        f"input={signature['input_dim']}, "
        f"hidden={signature['hidden_units']}, "
        f"output={signature['output_dim']}"
    )


def _infer_sensor_config_for_observation_dim(
    config: AppConfig,
    observation_dim: int,
) -> AppConfig | None:
    inferred_beam_count = int(observation_dim) - 3
    if inferred_beam_count < 3 or inferred_beam_count % 2 == 0:
        return None
    reconciled = copy.deepcopy(config)
    reconciled.sensors.lidar_beam_count = inferred_beam_count
    if len(build_observation_names(reconciled.sensors)) != observation_dim:
        return None
    return reconciled


def reconcile_checkpoint_config(
    config: AppConfig,
    state_dicts: dict[str, dict[str, torch.Tensor] | None],
) -> tuple[AppConfig, list[str]]:
    populated_state_dicts = [
        (label, state_dict)
        for label, state_dict in state_dicts.items()
        if state_dict
    ]
    if not populated_state_dicts:
        return copy.deepcopy(config), []

    signatures = [
        (label, infer_network_signature_from_state_dict(state_dict))
        for label, state_dict in populated_state_dicts
    ]
    reference_label, reference_signature = signatures[0]
    mismatched = [
        f"{label}({_signature_text(signature)})"
        for label, signature in signatures[1:]
        if signature != reference_signature
    ]
    if mismatched:
        raise ValueError(
            "Checkpoint contains mixed brain architectures: "
            f"{reference_label}({_signature_text(reference_signature)}) vs "
            + ", ".join(mismatched)
        )

    reconciled = copy.deepcopy(config)
    warnings: list[str] = []
    expected_input_dim = len(build_observation_names(reconciled.sensors))
    if expected_input_dim != reference_signature["input_dim"]:
        inferred_config = _infer_sensor_config_for_observation_dim(
            reconciled,
            reference_signature["input_dim"],
        )
        if inferred_config is None:
            raise ValueError(
                "Checkpoint observation size does not match the saved sensor "
                f"configuration ({expected_input_dim} vs "
                f"{reference_signature['input_dim']})."
            )
        warnings.append(
            "Checkpoint lidar beam count was reconciled from "
            f"{reconciled.sensors.lidar_beam_count} to "
            f"{inferred_config.sensors.lidar_beam_count} to match the saved "
            "weights."
        )
        reconciled = inferred_config

    if reference_signature["output_dim"] != len(ACTION_NAMES):
        raise ValueError(
            "Checkpoint action size does not match this project "
            f"({reference_signature['output_dim']} vs {len(ACTION_NAMES)})."
        )

    inferred_network = infer_network_config_from_state_dict(
        populated_state_dicts[0][1],
        fallback_network=reconciled.network,
    )
    if _network_units(reconciled.network) != _network_units(inferred_network):
        warnings.append(
            "Checkpoint network layout was reconciled from "
            f"{_network_units(reconciled.network)} to "
            f"{_network_units(inferred_network)} to match the saved weights."
        )
        reconciled.network = inferred_network

    return reconciled, warnings


class TrainerSession:
    def __init__(
        self,
        config: AppConfig,
        device: str | torch.device | None = None,
        initial_history: list[GenerationReport] | None = None,
        initial_state_dict: dict[str, torch.Tensor] | None = None,
        initial_best_state_dict: dict[str, torch.Tensor] | None = None,
        initial_best_metrics: dict[str, float] | None = None,
        initial_optimizer_state_dict: dict[str, Any] | None = None,
        initial_observation_normalizer_state: dict[str, Any] | None = None,
        initial_best_observation_normalizer_state: dict[str, Any] | None = None,
    ) -> None:
        self.config = copy.deepcopy(config)
        self.device_info = resolve_torch_device(device)
        self.device = self.device_info.device
        self.observation_names = build_observation_names(self.config.sensors)
        np.random.seed(self.config.ppo.seed)
        seed_torch_backends(self.config.ppo.seed)

        try:
            self.model = ActorCritic(
                observation_dim=len(self.observation_names),
                action_dim=len(ACTION_NAMES),
                network_config=self.config.network,
                init_std=self.config.ppo.init_std,
            ).to(self.device)
        except Exception as exc:
            if self.device.type == "cpu":
                raise
            self.device_info = _cpu_device_info(
                self.device_info.warnings
                + [f"{self.device_info.description} initialization failed: {exc}. Falling back to CPU."]
            )
            self.device = self.device_info.device
            self.model = ActorCritic(
                observation_dim=len(self.observation_names),
                action_dim=len(ACTION_NAMES),
                network_config=self.config.network,
                init_std=self.config.ppo.init_std,
            ).to(self.device)
        if initial_state_dict is not None:
            self.model.load_state_dict(initial_state_dict)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.ppo.learning_rate,
        )
        if initial_optimizer_state_dict:
            try:
                self.optimizer.load_state_dict(initial_optimizer_state_dict)
            except Exception:
                pass

        self.history = [
            GenerationReport.from_dict(report.to_dict())
            for report in (initial_history or [])
        ]
        self.generation_offset = len(self.history)
        self.stop_requested = False
        self.pause_requested = False
        self.parallel_env_count = max(
            1,
            min(
                self.config.ppo.games_per_generation,
                self.config.ppo.parallel_envs,
            ),
        )
        self.observation_normalizer = ObservationNormalizer(
            dimension=len(self.observation_names),
            state=initial_observation_normalizer_state,
        )
        self.best_state_dict = clone_state_dict(
            initial_best_state_dict or self.model.state_dict()
        )
        self.best_observation_normalizer_state = clone_normalizer_state(
            initial_best_observation_normalizer_state
            or self.observation_normalizer.state_dict()
        )
        metrics = dict(initial_best_metrics or {})
        self.best_metrics = {
            "finish_rate": metric_value(metrics, "finish_rate"),
            "mean_score": metric_value(metrics, "mean_score"),
        }

    def request_stop(self) -> None:
        self.stop_requested = True

    def request_pause(self) -> None:
        self.pause_requested = True

    def train(
        self,
        generations: int | None = None,
        on_generation: Callable[[GenerationReport], None] | None = None,
    ) -> TrainingSummary:
        target_generations = generations or self.config.ppo.target_generations
        status = "completed"

        for generation_index in range(
            self.generation_offset,
            self.generation_offset + target_generations,
        ):
            if self.stop_requested:
                status = "stopped"
                break

            report = self._run_generation(generation_index)
            metrics = {
                "finish_rate": report.finish_rate,
                "mean_score": report.mean_score,
            }
            if (not report.partial_generation) and is_better_candidate(
                metrics,
                self.best_metrics,
            ):
                self.best_metrics = metrics
                self.best_state_dict = clone_state_dict(self.model.state_dict())
                self.best_observation_normalizer_state = clone_normalizer_state(
                    self.observation_normalizer.state_dict()
                )
                report.best_updated = True

            report.current_state_dict = clone_state_dict(self.model.state_dict())
            report.best_state_dict = clone_state_dict(self.best_state_dict)
            report.current_normalizer_state = clone_normalizer_state(
                self.observation_normalizer.state_dict()
            )
            report.best_normalizer_state = clone_normalizer_state(
                self.best_observation_normalizer_state
            )
            self.history.append(report)
            if on_generation is not None:
                on_generation(report)

            if self.stop_requested:
                status = "stopped"
                break
            if self.pause_requested:
                status = "paused"
                break

        return TrainingSummary(
            status=status,
            history=list(self.history),
            current_state_dict=clone_state_dict(self.model.state_dict()),
            best_state_dict=clone_state_dict(self.best_state_dict),
            best_metrics=dict(self.best_metrics),
            current_optimizer_state_dict=clone_optimizer_state_dict(
                self.optimizer.state_dict()
            ),
            current_normalizer_state=clone_normalizer_state(
                self.observation_normalizer.state_dict()
            ),
            best_normalizer_state=clone_normalizer_state(
                self.best_observation_normalizer_state
            ),
            checkpoint_metadata=build_checkpoint_metadata(
                history=self.history,
                best_metrics=self.best_metrics,
                source_label="best",
                observation_normalizer_state=self.best_observation_normalizer_state,
                observation_names=self.observation_names,
                has_resume_state=True,
            ),
        )

    def _run_generation(self, generation_index: int) -> GenerationReport:
        buffer = RolloutBuffer()
        scores: list[float] = []
        distances: list[float] = []
        finished_episodes = 0
        total_steps = 0
        generation_seed_offset = self.config.ppo.seed + generation_index * 10000
        target_episodes = self.config.ppo.games_per_generation
        environments = [
            CarDriverEnv(
                physics=self.config.physics,
                rewards=self.config.rewards,
                sensors=self.config.sensors,
                track=self.config.track,
                seed=generation_seed_offset + slot_index,
            )
            for slot_index in range(self.parallel_env_count)
        ]
        active_episodes: list[ActiveEpisode | None] = [None] * len(environments)
        episodes_started = 0

        def launch_episode(slot_index: int) -> None:
            nonlocal episodes_started
            if episodes_started >= target_episodes:
                return
            episode_seed = generation_seed_offset + episodes_started
            raw_observation = environments[slot_index].reset(
                seed=episode_seed,
                dramatic=True,
            )
            active_episodes[slot_index] = ActiveEpisode(
                env=environments[slot_index],
                raw_observation=np.asarray(raw_observation, dtype=np.float32),
            )
            episodes_started += 1

        for slot_index in range(len(active_episodes)):
            launch_episode(slot_index)

        while any(episode is not None for episode in active_episodes):
            active_indices = [
                index
                for index, episode in enumerate(active_episodes)
                if episode is not None
            ]
            raw_batch = np.stack(
                [
                    np.asarray(active_episodes[index].raw_observation, dtype=np.float32)
                    for index in active_indices
                ],
                axis=0,
            )
            self.observation_normalizer.update(raw_batch)
            normalized_batch = self.observation_normalizer.normalize(raw_batch)

            observation_tensor = torch.as_tensor(
                normalized_batch,
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                action_tensor, log_prob_tensor, value_tensor, _ = self.model.act(
                    observation_tensor,
                    deterministic=False,
                )

            actions = action_tensor.cpu().numpy()
            log_probs = log_prob_tensor.cpu().numpy()
            values = value_tensor.cpu().numpy()

            for batch_index, slot_index in enumerate(active_indices):
                active_episode = active_episodes[slot_index]
                assert active_episode is not None

                next_observation, reward, done, info = active_episode.env.step(
                    actions[batch_index]
                )
                active_episode.trajectory.add(
                    observation=normalized_batch[batch_index],
                    action=actions[batch_index],
                    reward=reward,
                    log_prob=float(log_probs[batch_index]),
                    value=float(values[batch_index]),
                )
                total_steps += 1
                active_episode.raw_observation = np.asarray(
                    next_observation,
                    dtype=np.float32,
                )

                if not done:
                    continue

                active_episode.trajectory.final_info = dict(info)
                buffer.add_episode(active_episode.trajectory)
                scores.append(active_episode.trajectory.score)
                distances.append(float(info.get("distance_travelled", 0.0)))
                finished_episodes += int(bool(info.get("finished", False)))
                active_episodes[slot_index] = None

                if episodes_started < target_episodes and not self.stop_requested:
                    launch_episode(slot_index)

        if len(buffer) == 0:
            return GenerationReport(
                generation_index=generation_index,
                episodes_run=0,
                finish_rate=0.0,
                best_score=0.0,
                mean_score=0.0,
                mean_distance=0.0,
                total_steps=0,
                course_length=self.config.track.target_course_length,
            )

        optimization_stats = self._update_policy(buffer)
        episodes_run = max(1, len(scores))
        return GenerationReport(
            generation_index=generation_index,
            episodes_run=episodes_run,
            finish_rate=finished_episodes / episodes_run,
            best_score=max(scores),
            mean_score=float(np.mean(scores)),
            mean_distance=float(np.mean(distances or [0.0])),
            scores=scores,
            distances=distances,
            finished_episodes=finished_episodes,
            total_steps=total_steps,
            partial_generation=episodes_run < target_episodes,
            policy_loss=optimization_stats["policy_loss"],
            value_loss=optimization_stats["value_loss"],
            entropy=optimization_stats["entropy"],
            course_length=self.config.track.target_course_length,
        )

    def _update_policy(self, buffer: RolloutBuffer) -> dict[str, float]:
        batch = buffer.as_tensors(
            gamma=self.config.ppo.gamma,
            gae_lambda=self.config.ppo.gae_lambda,
            device=self.device,
        )
        advantages = batch["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        observations = batch["observations"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        returns = batch["returns"]
        batch_size = observations.shape[0]
        minibatch_size = min(self.config.ppo.minibatch_size, batch_size)

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
        }
        updates = 0

        for _ in range(self.config.ppo.ppo_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                new_log_probs, entropy, values = self.model.evaluate_actions(
                    observations[indices],
                    actions[indices],
                )
                ratio = (new_log_probs - old_log_probs[indices]).exp()
                unclipped = ratio * advantages[indices]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.ppo.clip_range,
                    1.0 + self.config.ppo.clip_range,
                ) * advantages[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(values, returns[indices])
                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.ppo.value_coef * value_loss
                    - self.config.ppo.entropy_coef * entropy_bonus
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.ppo.max_grad_norm,
                )
                self.optimizer.step()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy_bonus.item())
                updates += 1

        if updates:
            for key in metrics:
                metrics[key] /= updates
        return metrics


def build_policy_from_state(
    config: AppConfig,
    state_dict: dict[str, torch.Tensor] | None = None,
    device: str | torch.device | None = None,
) -> ActorCritic:
    device_info = resolve_torch_device(device)
    try:
        policy = ActorCritic(
            observation_dim=len(build_observation_names(config.sensors)),
            action_dim=len(ACTION_NAMES),
            network_config=config.network,
            init_std=config.ppo.init_std,
        ).to(device_info.device)
    except Exception as exc:
        if device_info.device.type == "cpu":
            raise
        policy = ActorCritic(
            observation_dim=len(build_observation_names(config.sensors)),
            action_dim=len(ACTION_NAMES),
            network_config=config.network,
            init_std=config.ppo.init_std,
        ).to("cpu")
    if state_dict is not None:
        try:
            policy.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                "Saved brain weights do not match the configured network shape. "
                "Reload the checkpoint so its config can be reconciled, or reset "
                "the brain if you intentionally changed the architecture."
            ) from exc
    policy.eval()
    return policy


def build_checkpoint_metadata(
    history: list[GenerationReport],
    best_metrics: dict[str, float],
    source_label: str,
    observation_normalizer_state: dict[str, Any] | None,
    observation_names: list[str],
    has_resume_state: bool = False,
) -> dict[str, Any]:
    best_report: GenerationReport | None = None
    best_candidate = {
        "finish_rate": -1.0,
        "mean_score": float("-inf"),
    }
    for report in history:
        candidate = {
            "finish_rate": report.finish_rate,
            "mean_score": report.mean_score,
        }
        if (not report.partial_generation) and is_better_candidate(
            candidate,
            best_candidate,
        ):
            best_report = report
            best_candidate = candidate

    last_report = history[-1] if history else None
    return {
        "schema_version": 3,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_label": source_label,
        "generation_count": len(history),
        "total_episodes": int(sum(report.episodes_run for report in history)),
        "total_steps": int(sum(report.total_steps for report in history)),
        "best_metrics": dict(best_metrics),
        "last_report": last_report.to_dict() if last_report is not None else None,
        "best_report": best_report.to_dict() if best_report is not None else None,
        "resume_supported": bool(has_resume_state),
        "observation_names": list(observation_names),
        "observation_normalizer_state": clone_normalizer_state(
            observation_normalizer_state
        ),
    }


def save_brain_checkpoint(
    path: Path | str,
    config: AppConfig,
    state_dict: dict[str, torch.Tensor],
    best_metrics: dict[str, float],
    history: list[GenerationReport],
    source_label: str,
    current_state_dict: dict[str, torch.Tensor] | None = None,
    best_state_dict: dict[str, torch.Tensor] | None = None,
    optimizer_state_dict: dict[str, Any] | None = None,
    observation_normalizer_state: dict[str, Any] | None = None,
    best_observation_normalizer_state: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    reconciled_config, _ = reconcile_checkpoint_config(
        config,
        {
            "state_dict": state_dict,
            "current_state_dict": current_state_dict,
            "best_state_dict": best_state_dict,
        },
    )
    observation_names = build_observation_names(reconciled_config.sensors)
    checkpoint_metadata = metadata or build_checkpoint_metadata(
        history=history,
        best_metrics=best_metrics,
        source_label=source_label,
        observation_normalizer_state=observation_normalizer_state,
        observation_names=observation_names,
        has_resume_state=bool(current_state_dict or optimizer_state_dict),
    )
    payload = {
        "config": reconciled_config.to_dict(),
        "state_dict": clone_state_dict(state_dict),
        "current_state_dict": clone_state_dict(current_state_dict or state_dict),
        "best_state_dict": clone_state_dict(best_state_dict or state_dict),
        "optimizer_state_dict": clone_optimizer_state_dict(optimizer_state_dict),
        "best_metrics": dict(best_metrics),
        "history": [report.to_dict() for report in history],
        "observation_names": list(observation_names),
        "action_names": list(ACTION_NAMES),
        "source_label": source_label,
        "observation_normalizer_state": clone_normalizer_state(
            observation_normalizer_state
        ),
        "best_observation_normalizer_state": clone_normalizer_state(
            best_observation_normalizer_state or observation_normalizer_state
        ),
        "metadata": checkpoint_metadata,
    }
    torch.save(payload, Path(path))


def load_brain_checkpoint(path: Path | str) -> dict[str, Any]:
    raw = torch.load(Path(path), map_location="cpu", weights_only=False)
    raw_config = AppConfig.from_dict(raw["config"])
    current_state_dict = clone_state_dict(raw.get("current_state_dict", raw["state_dict"]))
    best_state_dict = clone_state_dict(raw.get("best_state_dict", raw["state_dict"]))
    config, warnings = reconcile_checkpoint_config(
        raw_config,
        {
            "state_dict": raw["state_dict"],
            "current_state_dict": current_state_dict,
            "best_state_dict": best_state_dict,
        },
    )
    history = [GenerationReport.from_dict(item) for item in raw.get("history", [])]
    return {
        "config": config,
        "state_dict": raw["state_dict"],
        "current_state_dict": current_state_dict,
        "best_state_dict": best_state_dict,
        "optimizer_state_dict": clone_optimizer_state_dict(
            raw.get("optimizer_state_dict", {})
        ),
        "best_metrics": {
            "finish_rate": metric_value(raw.get("best_metrics", {}), "finish_rate"),
            "mean_score": metric_value(raw.get("best_metrics", {}), "mean_score"),
        },
        "history": history,
        "observation_names": build_observation_names(config.sensors),
        "action_names": raw.get("action_names", list(ACTION_NAMES)),
        "source_label": raw.get("source_label", "best"),
        "observation_normalizer_state": clone_normalizer_state(
            raw.get("observation_normalizer_state", {})
        ),
        "best_observation_normalizer_state": clone_normalizer_state(
            raw.get(
                "best_observation_normalizer_state",
                raw.get("observation_normalizer_state", {}),
            )
        ),
        "metadata": dict(
            raw.get(
                "metadata",
                build_checkpoint_metadata(
                    history=history,
                    best_metrics={
                        "finish_rate": metric_value(
                            raw.get("best_metrics", {}),
                            "finish_rate",
                        ),
                        "mean_score": metric_value(
                            raw.get("best_metrics", {}),
                            "mean_score",
                        ),
                    },
                    source_label=raw.get("source_label", "best"),
                    observation_normalizer_state=raw.get(
                        "observation_normalizer_state",
                        {},
                    ),
                    observation_names=raw.get(
                        "observation_names",
                        build_observation_names(config.sensors),
                    ),
                    has_resume_state=bool(raw.get("current_state_dict")),
                ),
            )
        ),
        "path": str(path),
        "warnings": warnings,
    }


def smoke_test(seed: int = 11) -> tuple[bool, str]:
    try:
        config = AppConfig()
        config.ppo.target_generations = 2
        config.ppo.games_per_generation = 4
        config.ppo.ppo_epochs = 2
        config.ppo.minibatch_size = 64
        config.ppo.seed = seed
        config.track.target_course_length = 320.0
        config.track.max_straight_length = 60.0
        config.physics.max_steps = 240

        env = CarDriverEnv(
            config.physics,
            config.rewards,
            config.sensors,
            config.track,
            seed=seed,
        )
        observation = env.reset()
        if observation.shape[0] != len(build_observation_names(config.sensors)):
            return False, "Smoke test failed: unexpected observation size."

        for _ in range(10):
            observation, _, done, _ = env.step(
                np.array([0.6, 0.0, 0.0, 0.0], dtype=np.float32)
            )
            if done:
                observation = env.reset()

        session = TrainerSession(config=config, device="cpu")
        summary = session.train()
        if not summary.history:
            return False, "Smoke test failed: no training history was produced."

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "car_driver_best.pt"
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
                observation_normalizer_state=summary.best_normalizer_state,
                best_observation_normalizer_state=summary.best_normalizer_state,
            )
            payload = load_brain_checkpoint(checkpoint_path)
            if len(payload["history"]) != len(summary.history):
                return False, "Smoke test failed: checkpoint history mismatch."

        last_report = summary.history[-1]
        return (
            True,
            "Smoke test passed: "
            f"{len(summary.history)} generations, "
            f"finish_rate={last_report.finish_rate:.3f}, "
            f"mean_score={last_report.mean_score:.3f}",
        )
    except Exception as exc:  # pragma: no cover
        return False, f"Smoke test failed with exception: {exc}"
