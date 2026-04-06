from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ACTIVATION_OPTIONS = [
    "relu",
    "tanh",
    "elu",
    "leaky_relu",
    "selu",
    "identity",
]


@dataclass
class LayerConfig:
    units: int
    activation: str = "relu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayerConfig":
        return cls(
            units=int(data.get("units", 8)),
            activation=str(data.get("activation", "relu")),
        )


@dataclass
class PhysicsConfig:
    max_speed: float = 42.0
    acceleration: float = 8.5
    brake_deceleration: float = 14.0
    coasting_deceleration: float = 1.8
    steering_response_deg: float = 160.0
    max_steering_angle_deg: float = 30.0
    wheelbase: float = 2.7
    car_length: float = 4.4
    car_width: float = 1.85
    dt: float = 0.1
    max_steps: int = 2500
    stalled_timeout_seconds: float = 3.0
    stalled_speed_threshold: float = 0.75
    control_dead_zone: float = 0.1
    fuel_tank_liters: float = 55.0
    initial_fuel_liters: float = 55.0
    fuel_usage_lps: float = 0.08

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SensorConfig:
    lidar_range: float = 50.0
    lidar_beam_count: int = 13
    lidar_angle_spacing_deg: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrackConfig:
    target_course_length: float = 3000.0
    road_width: float = 10.0
    min_straight_length: float = 35.0
    max_straight_length: float = 140.0
    min_turn_radius: float = 20.0
    max_turn_radius: float = 90.0
    min_turn_angle_deg: float = 14.0
    max_turn_angle_deg: float = 72.0
    generation_seed_offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RewardConfig:
    finish_bonus: float = 420.0
    progress_reward_scale: float = 0.9
    centerline_bonus: float = 0.35
    heading_bonus: float = 0.25
    alive_bonus: float = 0.02
    step_penalty: float = 0.01
    steering_penalty: float = 0.03
    brake_penalty: float = 0.02
    crash_penalty: float = 70.0
    offroad_penalty: float = 110.0
    timeout_penalty: float = 55.0
    stall_penalty: float = 35.0
    fuel_efficiency_bonus: float = 0.02

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PPOConfig:
    target_generations: int = 150
    games_per_generation: int = 25
    parallel_envs: int = 8
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.97
    clip_range: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.7
    ppo_epochs: int = 8
    minibatch_size: int = 256
    max_grad_norm: float = 0.7
    init_std: float = 0.45
    seed: int = 7

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NetworkConfig:
    hidden_layers: list[LayerConfig] = field(
        default_factory=lambda: [
            LayerConfig(12, "tanh"),
            LayerConfig(60, "relu"),
            LayerConfig(10, "relu"),
        ]
    )
    output_activation: str = "tanh"

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_layers": [layer.to_dict() for layer in self.hidden_layers],
            "output_activation": self.output_activation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NetworkConfig":
        layers = [
            LayerConfig.from_dict(item)
            for item in data.get("hidden_layers", [])
        ]
        if not layers:
            layers = [
                LayerConfig(12, "tanh"),
                LayerConfig(60, "relu"),
                LayerConfig(10, "relu"),
            ]
        return cls(
            hidden_layers=layers,
            output_activation=str(data.get("output_activation", "tanh")),
        )


@dataclass
class AppConfig:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "physics": self.physics.to_dict(),
            "sensors": self.sensors.to_dict(),
            "track": self.track.to_dict(),
            "rewards": self.rewards.to_dict(),
            "ppo": self.ppo.to_dict(),
            "network": self.network.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        return cls(
            physics=PhysicsConfig(**data.get("physics", {})),
            sensors=SensorConfig(**data.get("sensors", {})),
            track=TrackConfig(**data.get("track", {})),
            rewards=RewardConfig(**data.get("rewards", {})),
            ppo=PPOConfig(**data.get("ppo", {})),
            network=NetworkConfig.from_dict(data.get("network", {})),
        )
