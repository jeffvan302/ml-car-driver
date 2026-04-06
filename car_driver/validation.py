from __future__ import annotations

import math
from dataclasses import dataclass, field

from car_driver.config import ACTIVATION_OPTIONS, AppConfig


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_app_config(config: AppConfig) -> ValidationResult:
    result = ValidationResult()
    physics = config.physics
    sensors = config.sensors
    track = config.track
    rewards = config.rewards
    ppo = config.ppo
    network = config.network

    if physics.max_speed <= 0.0:
        result.errors.append("Max speed must be positive.")
    if physics.acceleration <= 0.0:
        result.errors.append("Acceleration must be positive.")
    if physics.brake_deceleration <= 0.0:
        result.errors.append("Brake deceleration must be positive.")
    if physics.coasting_deceleration <= 0.0:
        result.errors.append("Coasting deceleration must be positive.")
    if physics.steering_response_deg <= 0.0:
        result.errors.append("Steering response must be positive.")
    if physics.max_steering_angle_deg <= 0.0 or physics.max_steering_angle_deg >= 89.0:
        result.errors.append("Max steering angle must be between 0 and 89 degrees.")
    if physics.wheelbase <= 0.0:
        result.errors.append("Wheelbase must be positive.")
    if physics.car_length <= 0.0 or physics.car_width <= 0.0:
        result.errors.append("Car dimensions must be positive.")
    if physics.dt <= 0.0:
        result.errors.append("Physics dt must be positive.")
    if physics.max_steps < 25:
        result.errors.append("Max steps must be at least 25.")
    if physics.stalled_timeout_seconds <= 0.0:
        result.errors.append("Stalled timeout must be positive.")
    if physics.stalled_speed_threshold < 0.0:
        result.errors.append("Stalled speed threshold cannot be negative.")
    if not 0.0 <= physics.control_dead_zone < 1.0:
        result.errors.append("Control dead zone must be in the range [0, 1).")
    if physics.fuel_tank_liters <= 0.0:
        result.errors.append("Fuel tank volume must be positive.")
    if physics.initial_fuel_liters < 0.0:
        result.errors.append("Initial fuel cannot be negative.")
    if physics.initial_fuel_liters > physics.fuel_tank_liters:
        result.errors.append("Initial fuel cannot exceed the fuel tank volume.")
    if physics.fuel_usage_lps < 0.0:
        result.errors.append("Fuel usage cannot be negative.")

    if sensors.lidar_range <= 0.0:
        result.errors.append("Lidar range must be positive.")
    if sensors.lidar_beam_count < 3:
        result.errors.append("Lidar beam count must be at least 3.")
    if sensors.lidar_beam_count % 2 == 0:
        result.errors.append("Lidar beam count must be odd so one beam points straight ahead.")
    if sensors.lidar_angle_spacing_deg <= 0.0:
        result.errors.append("Lidar angle spacing must be positive.")

    if track.target_course_length <= 50.0:
        result.errors.append("Target course length must be greater than 50 meters.")
    if track.road_width <= physics.car_width:
        result.errors.append("Road width must be wider than the car width.")
    if track.min_straight_length <= 0.0 or track.max_straight_length <= 0.0:
        result.errors.append("Straight segment lengths must be positive.")
    if track.min_straight_length > track.max_straight_length:
        result.errors.append("Minimum straight length must be less than or equal to the maximum straight length.")
    if track.min_turn_radius <= 0.0 or track.max_turn_radius <= 0.0:
        result.errors.append("Turn radii must be positive.")
    if track.min_turn_radius > track.max_turn_radius:
        result.errors.append("Minimum turn radius must be less than or equal to the maximum turn radius.")
    if track.min_turn_angle_deg <= 0.0 or track.max_turn_angle_deg <= 0.0:
        result.errors.append("Turn angles must be positive.")
    if track.min_turn_angle_deg > track.max_turn_angle_deg:
        result.errors.append("Minimum turn angle must be less than or equal to the maximum turn angle.")
    if track.max_turn_angle_deg >= 179.0:
        result.errors.append("Maximum turn angle must stay below 179 degrees.")

    if ppo.target_generations < 1:
        result.errors.append("Target generations must be at least 1.")
    if ppo.games_per_generation < 1:
        result.errors.append("Games per generation must be at least 1.")
    if ppo.parallel_envs < 1:
        result.errors.append("Parallel envs must be at least 1.")
    if ppo.learning_rate <= 0.0:
        result.errors.append("Learning rate must be positive.")
    if not 0.0 < ppo.gamma <= 0.999999:
        result.errors.append("Gamma must be in the range (0, 0.999999].")
    if not 0.0 < ppo.gae_lambda <= 0.999999:
        result.errors.append("GAE lambda must be in the range (0, 0.999999].")
    if not 0.0 < ppo.clip_range <= 1.0:
        result.errors.append("Clip range must be in the range (0, 1].")
    if ppo.entropy_coef < 0.0:
        result.errors.append("Entropy coef cannot be negative.")
    if ppo.value_coef < 0.0:
        result.errors.append("Value coef cannot be negative.")
    if ppo.ppo_epochs < 1:
        result.errors.append("PPO epochs must be at least 1.")
    if ppo.minibatch_size < 8:
        result.errors.append("Minibatch size must be at least 8.")
    if ppo.max_grad_norm <= 0.0:
        result.errors.append("Max grad norm must be positive.")
    if ppo.init_std <= 0.0:
        result.errors.append("Action std must be positive.")
    if ppo.parallel_envs > ppo.games_per_generation:
        result.warnings.append("Parallel envs is larger than games per generation, so it will be capped to the generation size.")

    if not network.hidden_layers:
        result.errors.append("At least one hidden layer is required.")
    for index, layer in enumerate(network.hidden_layers, start=1):
        if layer.units < 1:
            result.errors.append(f"Hidden layer {index} must have at least one unit.")
        if layer.activation not in ACTIVATION_OPTIONS:
            result.errors.append(
                f"Hidden layer {index} uses unsupported activation '{layer.activation}'."
            )
    if network.output_activation not in ACTIVATION_OPTIONS:
        result.errors.append(
            f"Output activation '{network.output_activation}' is not supported."
        )

    for name, value in rewards.to_dict().items():
        if not math.isfinite(float(value)):
            result.errors.append(f"Reward value '{name}' must be finite.")

    if track.road_width <= physics.car_width * 1.35:
        result.warnings.append("Road width is only slightly wider than the car, so early training may be unforgiving.")
    if track.min_turn_radius <= physics.wheelbase * 3.0:
        result.warnings.append("Minimum turn radius is very tight relative to the wheelbase.")
    if sensors.lidar_range < track.road_width * 1.5:
        result.warnings.append("Lidar range is short relative to the road width, which limits look-ahead.")
    if sensors.lidar_range > track.max_straight_length * 2.5:
        result.warnings.append("Lidar range is very long relative to the straight lengths and may add noisy far-field input.")
    if physics.acceleration <= physics.coasting_deceleration * 1.2:
        result.warnings.append("Acceleration is only slightly stronger than coasting loss, so the car may struggle to build speed.")
    if physics.brake_deceleration <= physics.acceleration:
        result.warnings.append("Brake deceleration is not stronger than acceleration, so stopping may feel weak.")
    if rewards.offroad_penalty <= 0.0:
        result.warnings.append("Off-road penalty is not positive, so the agent may not strongly avoid leaving the road.")
    if rewards.timeout_penalty <= 0.0:
        result.warnings.append("Timeout penalty is not positive, so indecisive driving may persist.")
    if rewards.stall_penalty <= 0.0:
        result.warnings.append("Stall penalty is not positive, so stopping on track may not be discouraged.")

    return result
