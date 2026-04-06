from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from car_driver.config import PhysicsConfig, RewardConfig, SensorConfig, TrackConfig


def wrap_angle(angle: float) -> float:
    return ((angle + math.pi) % (2.0 * math.pi)) - math.pi


def heading_vector(angle: float) -> np.ndarray:
    return np.array([math.sin(angle), math.cos(angle)], dtype=np.float32)


def right_vector(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), -math.sin(angle)], dtype=np.float32)


def lidar_angle_offsets(sensor_config: SensorConfig) -> list[float]:
    half = sensor_config.lidar_beam_count // 2
    start = -half * sensor_config.lidar_angle_spacing_deg
    return [
        start + index * sensor_config.lidar_angle_spacing_deg
        for index in range(sensor_config.lidar_beam_count)
    ]


def build_observation_names(
    sensor_config: SensorConfig | None = None,
) -> list[str]:
    sensor_config = sensor_config or SensorConfig()
    names = ["speed_norm"]
    for angle in lidar_angle_offsets(sensor_config):
        names.append(f"lidar_{int(round(angle)):+d}deg")
    names.extend(
        [
            "heading_error",
            "lateral_offset",
        ]
    )
    return names


OBSERVATION_NAMES = tuple(build_observation_names())


@dataclass
class TrackGeometry:
    centerline: np.ndarray
    left_boundary: np.ndarray
    right_boundary: np.ndarray
    headings: np.ndarray
    normals: np.ndarray
    cumulative_lengths: np.ndarray
    road_width: float
    target_length: float
    point_spacing: float
    generation_seed: int


@dataclass
class CarState:
    x: float
    y: float
    heading: float
    speed: float
    fuel: float
    steering_angle: float = 0.0
    gas: float = 0.0
    brake: float = 0.0
    steer_signal: float = 0.0
    steps: int = 0
    elapsed_time: float = 0.0
    stalled_time: float = 0.0
    score: float = 0.0
    progress_distance: float = 0.0
    last_progress_distance: float = 0.0
    heading_error: float = 0.0
    lateral_offset: float = 0.0
    nearest_index: int = 0
    lidar_values: np.ndarray = field(
        default_factory=lambda: np.ones(13, dtype=np.float32)
    )
    lidar_hits: np.ndarray = field(
        default_factory=lambda: np.zeros((13, 2), dtype=np.float32)
    )
    trail: list[tuple[float, float]] = field(default_factory=list)


class CarDriverEnv:
    def __init__(
        self,
        physics: PhysicsConfig | None = None,
        rewards: RewardConfig | None = None,
        sensors: SensorConfig | None = None,
        track: TrackConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.physics = physics or PhysicsConfig()
        self.rewards = rewards or RewardConfig()
        self.sensors = sensors or SensorConfig()
        self.track_config = track or TrackConfig()
        self.rng = random.Random(seed)
        self.track: TrackGeometry | None = None
        self.state: CarState | None = None
        self.active_track_seed = seed or 0
        self.last_info: dict[str, Any] = {}

    @property
    def observation_names(self) -> list[str]:
        return build_observation_names(self.sensors)

    @property
    def observation_size(self) -> int:
        return len(self.observation_names)

    @property
    def action_size(self) -> int:
        return 4

    def set_physics(self, physics: PhysicsConfig) -> None:
        self.physics = physics

    def set_rewards(self, rewards: RewardConfig) -> None:
        self.rewards = rewards

    def set_sensors(self, sensors: SensorConfig) -> None:
        self.sensors = sensors

    def set_track(self, track: TrackConfig) -> None:
        self.track_config = track

    def reset(self, seed: int | None = None, dramatic: bool = True) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)
            self.active_track_seed = seed + self.track_config.generation_seed_offset
        else:
            self.active_track_seed = self.rng.randint(0, 2**31 - 1)

        self.track = self._generate_track(self.active_track_seed)
        assert self.track is not None

        start_heading = float(self.track.headings[0])
        lateral_range = self.track.road_width * (0.06 if dramatic else 0.03)
        lateral_offset = self.rng.uniform(-lateral_range, lateral_range)
        heading_jitter = math.radians(self.rng.uniform(-4.0, 4.0 if dramatic else 2.0))
        spawn = self.track.centerline[0] + self.track.normals[0] * lateral_offset
        initial_fuel = min(
            self.physics.initial_fuel_liters,
            self.physics.fuel_tank_liters,
        )
        self.state = CarState(
            x=float(spawn[0]),
            y=float(spawn[1]),
            heading=wrap_angle(start_heading + heading_jitter),
            speed=0.0,
            fuel=max(initial_fuel, 0.0),
            nearest_index=0,
        )
        self.state.trail.append((self.state.x, self.state.y))
        self._update_track_state()
        self.last_info = {
            "event": "running",
            "finished": False,
            "crashed": False,
            "offroad": False,
            "timeout": False,
            "stalled": False,
            "distance_travelled": 0.0,
            "elapsed_time": 0.0,
            "score": 0.0,
            "track_seed": self.active_track_seed,
        }
        return self._get_observation()

    def step(
        self,
        action: np.ndarray | list[float] | tuple[float, float, float, float],
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        assert self.state is not None
        assert self.track is not None

        gas, brake, steer_signal = self._resolve_action(action)
        dt = self.physics.dt

        max_steer = math.radians(self.physics.max_steering_angle_deg)
        steering_rate = math.radians(self.physics.steering_response_deg)
        target_steering = steer_signal * max_steer
        steering_delta = float(
            np.clip(
                target_steering - self.state.steering_angle,
                -steering_rate * dt,
                steering_rate * dt,
            )
        )
        self.state.steering_angle = float(
            np.clip(self.state.steering_angle + steering_delta, -max_steer, max_steer)
        )

        gas_effective = 0.0
        if gas > 0.0 and self.state.fuel > 0.0:
            requested_fuel = gas * self.physics.fuel_usage_lps * dt
            fuel_used = min(requested_fuel, self.state.fuel)
            if requested_fuel > 1e-8:
                gas_effective = gas * (fuel_used / requested_fuel)
            self.state.fuel = max(0.0, self.state.fuel - fuel_used)

        acceleration = gas_effective * self.physics.acceleration
        if brake > 0.0:
            acceleration -= brake * self.physics.brake_deceleration
        elif gas_effective <= 0.0:
            acceleration -= self.physics.coasting_deceleration

        self.state.speed = float(
            np.clip(
                self.state.speed + acceleration * dt,
                0.0,
                self.physics.max_speed,
            )
        )
        if self.state.speed > 1e-5:
            yaw_rate = (
                self.state.speed
                / max(self.physics.wheelbase, 1e-6)
                * math.tan(self.state.steering_angle)
            )
        else:
            yaw_rate = 0.0
        self.state.heading = wrap_angle(self.state.heading + yaw_rate * dt)

        forward = heading_vector(self.state.heading)
        self.state.x += float(forward[0] * self.state.speed * dt)
        self.state.y += float(forward[1] * self.state.speed * dt)
        self.state.steps += 1
        self.state.elapsed_time += dt
        self.state.gas = gas_effective
        self.state.brake = brake
        self.state.steer_signal = steer_signal
        if self.state.speed < self.physics.stalled_speed_threshold:
            self.state.stalled_time += dt
        else:
            self.state.stalled_time = 0.0
        self.state.trail.append((self.state.x, self.state.y))
        if len(self.state.trail) > 180:
            self.state.trail = self.state.trail[-180:]

        self._update_track_state()
        reward, done, info = self._calculate_reward_and_termination()
        self.state.score += reward
        info["score"] = self.state.score
        self.last_info = info
        return self._get_observation(), reward, done, info

    def _resolve_action(
        self,
        action: np.ndarray | list[float] | tuple[float, float, float, float],
    ) -> tuple[float, float, float]:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape[0] < 4:
            padded = np.zeros(4, dtype=np.float32)
            padded[: values.shape[0]] = values
            values = padded
        values = np.clip(values[:4], -1.0, 1.0)
        gas = max(float(values[0]), 0.0)
        brake = max(float(values[1]), 0.0)
        left = max(float(values[2]), 0.0)
        right = max(float(values[3]), 0.0)
        threshold = self.physics.control_dead_zone

        gas = gas if gas >= threshold else 0.0
        brake = brake if brake >= threshold else 0.0
        left = left if left >= threshold else 0.0
        right = right if right >= threshold else 0.0

        if brake > 0.0 and gas > 0.0:
            gas = 0.0
        if left > 0.0 and right > 0.0:
            steer_signal = 0.0
        elif left > 0.0:
            steer_signal = -left
        elif right > 0.0:
            steer_signal = right
        else:
            steer_signal = 0.0
        return gas, brake, steer_signal

    def _update_track_state(self) -> None:
        assert self.state is not None
        assert self.track is not None
        position = np.array([self.state.x, self.state.y], dtype=np.float32)
        located = self._locate_point(position, self.state.nearest_index)
        self.state.nearest_index = located["index"]
        self.state.progress_distance = min(
            located["progress_distance"],
            self.track.target_length,
        )
        self.state.heading_error = wrap_angle(self.state.heading - located["road_heading"])
        self.state.lateral_offset = located["lateral_offset"] / max(
            self.track.road_width * 0.5,
            1e-6,
        )
        lidar_values, lidar_hits = self._compute_lidar(
            position=position,
            heading=self.state.heading,
            nearest_index=self.state.nearest_index,
        )
        self.state.lidar_values = lidar_values
        self.state.lidar_hits = lidar_hits

    def _calculate_reward_and_termination(
        self,
    ) -> tuple[float, bool, dict[str, Any]]:
        assert self.state is not None
        assert self.track is not None

        progress_delta = self.state.progress_distance - self.state.last_progress_distance
        self.state.last_progress_distance = self.state.progress_distance

        center_score = max(0.0, 1.0 - min(abs(self.state.lateral_offset), 1.5))
        heading_score = max(0.0, math.cos(self.state.heading_error))
        steering_ratio = abs(self.state.steering_angle) / max(
            math.radians(self.physics.max_steering_angle_deg),
            1e-6,
        )
        fuel_ratio = self.state.fuel / max(self.physics.fuel_tank_liters, 1e-6)
        reward = 0.0
        reward += self.rewards.alive_bonus
        reward += self.rewards.progress_reward_scale * progress_delta
        reward += self.rewards.centerline_bonus * center_score * self.physics.dt
        reward += self.rewards.heading_bonus * heading_score * self.physics.dt
        reward += self.rewards.fuel_efficiency_bonus * fuel_ratio * self.physics.dt
        reward -= self.rewards.step_penalty
        reward -= self.rewards.steering_penalty * steering_ratio * self.physics.dt
        reward -= self.rewards.brake_penalty * self.state.brake * self.physics.dt
        if progress_delta < 0.0:
            reward += progress_delta * 0.5

        done = False
        finished = False
        crashed = False
        offroad = False
        timeout = False
        stalled = False
        event = "running"

        if self.state.progress_distance >= self.track.target_length - self.track.point_spacing:
            done = True
            finished = True
            event = "finished"
            reward += self.rewards.finish_bonus
        else:
            body_inside = self._body_inside_road()
            if not body_inside:
                done = True
                crashed = True
                offroad = True
                event = "crash"
                speed_ratio = self.state.speed / max(self.physics.max_speed, 1e-6)
                reward -= self.rewards.offroad_penalty
                reward -= self.rewards.crash_penalty * (0.4 + 0.6 * speed_ratio)
            elif self.state.steps >= self.physics.max_steps:
                done = True
                timeout = True
                event = "timeout"
                reward -= self.rewards.timeout_penalty
            elif self.state.stalled_time >= self.physics.stalled_timeout_seconds:
                done = True
                stalled = True
                event = "stall"
                reward -= self.rewards.stall_penalty

        info = {
            "event": event,
            "finished": finished,
            "crashed": crashed,
            "offroad": offroad,
            "timeout": timeout,
            "stalled": stalled,
            "distance_travelled": self.state.progress_distance,
            "elapsed_time": self.state.elapsed_time,
            "speed": self.state.speed,
            "fuel_ratio": fuel_ratio,
            "heading_error": self.state.heading_error,
            "lateral_offset": self.state.lateral_offset,
            "progress_ratio": self.state.progress_distance / max(self.track.target_length, 1e-6),
            "track_seed": self.active_track_seed,
        }
        return reward, done, info

    def _get_observation(self) -> np.ndarray:
        assert self.state is not None
        speed_norm = self.state.speed / max(self.physics.max_speed, 1e-6)
        observation = np.concatenate(
            [
                np.array([speed_norm], dtype=np.float32),
                np.asarray(self.state.lidar_values, dtype=np.float32),
                np.array(
                    [
                        self.state.heading_error / math.pi,
                        float(np.clip(self.state.lateral_offset, -1.5, 1.5)),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        return observation

    def _locate_point(
        self,
        point: np.ndarray,
        hint_index: int | None,
    ) -> dict[str, float | int]:
        assert self.track is not None
        index = self._nearest_track_index(point, hint_index)
        center = self.track.centerline[index]
        normal = self.track.normals[index]
        delta = point - center
        return {
            "index": index,
            "progress_distance": float(self.track.cumulative_lengths[index]),
            "road_heading": float(self.track.headings[index]),
            "lateral_offset": float(np.dot(delta, normal)),
        }

    def _nearest_track_index(
        self,
        point: np.ndarray,
        hint_index: int | None,
    ) -> int:
        assert self.track is not None
        points = self.track.centerline
        count = points.shape[0]
        if hint_index is None:
            deltas = points - point
            return int(np.argmin(np.sum(deltas * deltas, axis=1)))

        window = max(18, int(math.ceil(self.sensors.lidar_range / max(self.track.point_spacing, 1e-6))))
        start = max(0, hint_index - window // 2)
        stop = min(count, hint_index + window)
        local = points[start:stop]
        deltas = local - point
        local_index = int(np.argmin(np.sum(deltas * deltas, axis=1)))
        index = start + local_index
        local_distance_sq = float(np.sum((points[index] - point) ** 2))
        if index in (start, stop - 1) or local_distance_sq > (self.track.road_width * 3.5) ** 2:
            deltas = points - point
            index = int(np.argmin(np.sum(deltas * deltas, axis=1)))
        return index

    def _body_inside_road(self) -> bool:
        assert self.state is not None
        assert self.track is not None
        for corner in self._car_corners():
            located = self._locate_point(corner, self.state.nearest_index)
            if abs(float(located["lateral_offset"])) > self.track.road_width * 0.5:
                return False
        return True

    def _car_corners(self) -> list[np.ndarray]:
        assert self.state is not None
        forward = heading_vector(self.state.heading)
        right = right_vector(self.state.heading)
        center = np.array([self.state.x, self.state.y], dtype=np.float32)
        half_length = self.physics.car_length * 0.5
        half_width = self.physics.car_width * 0.5
        front = center + forward * half_length
        rear = center - forward * half_length
        return [
            front - right * half_width,
            front + right * half_width,
            rear + right * half_width,
            rear - right * half_width,
        ]

    def _compute_lidar(
        self,
        position: np.ndarray,
        heading: float,
        nearest_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.track is not None
        beam_values: list[float] = []
        beam_hits: list[np.ndarray] = []
        max_distance = self.sensors.lidar_range
        segment_window = max(
            24,
            int(math.ceil(max_distance / max(self.track.point_spacing, 1e-6))) + 6,
        )
        start = max(0, nearest_index - segment_window // 2)
        stop = min(self.track.centerline.shape[0] - 1, nearest_index + segment_window)
        left = self.track.left_boundary[start : stop + 1]
        right = self.track.right_boundary[start : stop + 1]

        for angle_deg in lidar_angle_offsets(self.sensors):
            ray_heading = heading + math.radians(angle_deg)
            ray_dir = heading_vector(ray_heading)
            best_distance = max_distance
            best_point = position + ray_dir * max_distance
            for boundary in (left, right):
                for index in range(boundary.shape[0] - 1):
                    distance = ray_segment_intersection(
                        position,
                        ray_dir,
                        boundary[index],
                        boundary[index + 1],
                    )
                    if distance is None or distance >= best_distance:
                        continue
                    best_distance = distance
                    best_point = position + ray_dir * distance
            beam_values.append(min(best_distance / max(max_distance, 1e-6), 1.0))
            beam_hits.append(best_point.astype(np.float32))
        return np.asarray(beam_values, dtype=np.float32), np.asarray(
            beam_hits,
            dtype=np.float32,
        )

    def _generate_track(self, seed: int) -> TrackGeometry:
        rng = random.Random(seed)
        spacing = 1.5
        last_track: TrackGeometry | None = None
        for attempt in range(10):
            center_points = [np.array([0.0, 0.0], dtype=np.float32)]
            headings = [0.0]
            cumulative = [0.0]
            total_length = 0.0
            current_heading = 0.0
            current_point = np.array([0.0, 0.0], dtype=np.float32)
            force_straight = True

            while total_length < self.track_config.target_course_length:
                remaining = self.track_config.target_course_length - total_length
                if force_straight or rng.random() < 0.42:
                    segment_length = min(
                        remaining,
                        rng.uniform(
                            self.track_config.min_straight_length,
                            self.track_config.max_straight_length,
                        ),
                    )
                    current_point, current_heading, total_length = append_straight_segment(
                        center_points=center_points,
                        headings=headings,
                        cumulative=cumulative,
                        start_point=current_point,
                        start_heading=current_heading,
                        length=segment_length,
                        step=spacing,
                        total_length=total_length,
                    )
                    force_straight = False
                    continue

                turn_direction = -1.0 if rng.random() < 0.5 else 1.0
                turn_radius = rng.uniform(
                    self.track_config.min_turn_radius,
                    self.track_config.max_turn_radius,
                )
                turn_angle_deg = rng.uniform(
                    self.track_config.min_turn_angle_deg,
                    self.track_config.max_turn_angle_deg,
                )
                arc_length = min(
                    remaining,
                    turn_radius * math.radians(turn_angle_deg),
                )
                if arc_length < spacing:
                    force_straight = True
                    continue
                current_point, current_heading, total_length = append_turn_segment(
                    center_points=center_points,
                    headings=headings,
                    cumulative=cumulative,
                    start_point=current_point,
                    start_heading=current_heading,
                    radius=turn_radius,
                    direction=turn_direction,
                    arc_length=arc_length,
                    step=spacing,
                    total_length=total_length,
                )
                force_straight = True

            centerline = np.asarray(center_points, dtype=np.float32)
            heading_array = np.asarray(headings, dtype=np.float32)
            normals = np.stack([right_vector(float(angle)) for angle in heading_array], axis=0)
            half_width = self.track_config.road_width * 0.5
            left_boundary = centerline - normals * half_width
            right_boundary = centerline + normals * half_width
            cumulative_lengths = np.asarray(cumulative, dtype=np.float32)
            last_track = TrackGeometry(
                centerline=centerline,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                headings=heading_array,
                normals=normals.astype(np.float32),
                cumulative_lengths=cumulative_lengths,
                road_width=self.track_config.road_width,
                target_length=float(cumulative_lengths[-1]),
                point_spacing=spacing,
                generation_seed=seed + attempt,
            )
            if not track_self_intersects(centerline, spacing):
                return last_track
        assert last_track is not None
        return last_track

    def snapshot(self) -> dict[str, Any]:
        if self.state is None or self.track is None:
            return {
                "physics": self.physics.to_dict(),
                "sensors": self.sensors.to_dict(),
                "track_config": self.track_config.to_dict(),
                "track": None,
                "state": None,
                "info": dict(self.last_info),
                "active_track_seed": self.active_track_seed,
            }
        return {
            "physics": self.physics.to_dict(),
            "sensors": self.sensors.to_dict(),
            "track_config": self.track_config.to_dict(),
            "track": {
                "centerline": self.track.centerline,
                "left_boundary": self.track.left_boundary,
                "right_boundary": self.track.right_boundary,
                "headings": self.track.headings,
                "cumulative_lengths": self.track.cumulative_lengths,
                "road_width": self.track.road_width,
                "target_length": self.track.target_length,
                "point_spacing": self.track.point_spacing,
            },
            "state": {
                "x": self.state.x,
                "y": self.state.y,
                "heading": self.state.heading,
                "speed": self.state.speed,
                "fuel": self.state.fuel,
                "steering_angle": self.state.steering_angle,
                "gas": self.state.gas,
                "brake": self.state.brake,
                "steer_signal": self.state.steer_signal,
                "steps": self.state.steps,
                "elapsed_time": self.state.elapsed_time,
                "stalled_time": self.state.stalled_time,
                "score": self.state.score,
                "progress_distance": self.state.progress_distance,
                "heading_error": self.state.heading_error,
                "lateral_offset": self.state.lateral_offset,
                "nearest_index": self.state.nearest_index,
                "lidar_values": self.state.lidar_values.copy(),
                "lidar_hits": self.state.lidar_hits.copy(),
                "trail": np.asarray(self.state.trail, dtype=np.float32),
            },
            "info": dict(self.last_info),
            "active_track_seed": self.active_track_seed,
        }


def append_straight_segment(
    center_points: list[np.ndarray],
    headings: list[float],
    cumulative: list[float],
    start_point: np.ndarray,
    start_heading: float,
    length: float,
    step: float,
    total_length: float,
) -> tuple[np.ndarray, float, float]:
    current_point = start_point.copy()
    current_heading = start_heading
    remaining = length
    while remaining > 1e-6:
        distance = min(step, remaining)
        current_point = current_point + heading_vector(current_heading) * distance
        total_length += distance
        center_points.append(current_point.copy())
        headings.append(current_heading)
        cumulative.append(total_length)
        remaining -= distance
    return current_point, current_heading, total_length


def append_turn_segment(
    center_points: list[np.ndarray],
    headings: list[float],
    cumulative: list[float],
    start_point: np.ndarray,
    start_heading: float,
    radius: float,
    direction: float,
    arc_length: float,
    step: float,
    total_length: float,
) -> tuple[np.ndarray, float, float]:
    current_point = start_point.copy()
    current_heading = start_heading
    remaining = arc_length
    while remaining > 1e-6:
        distance = min(step, remaining)
        heading_delta = direction * distance / max(radius, 1e-6)
        heading_mid = current_heading + 0.5 * heading_delta
        current_point = current_point + heading_vector(heading_mid) * distance
        current_heading = wrap_angle(current_heading + heading_delta)
        total_length += distance
        center_points.append(current_point.copy())
        headings.append(current_heading)
        cumulative.append(total_length)
        remaining -= distance
    return current_point, current_heading, total_length


def track_self_intersects(centerline: np.ndarray, spacing: float) -> bool:
    if centerline.shape[0] < 12:
        return False
    stride = max(1, int(round(6.0 / max(spacing, 1e-6))))
    sampled = centerline[::stride]
    if sampled.shape[0] < 6:
        return False
    for index in range(sampled.shape[0] - 1):
        a1 = sampled[index]
        a2 = sampled[index + 1]
        for other_index in range(index + 4, sampled.shape[0] - 1):
            b1 = sampled[other_index]
            b2 = sampled[other_index + 1]
            if segments_intersect(a1, a2, b1, b2):
                return True
    return False


def segments_intersect(
    a1: np.ndarray,
    a2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
) -> bool:
    def cross_2d(u: np.ndarray, v: np.ndarray) -> float:
        return float(u[0] * v[1] - u[1] * v[0])

    r = a2 - a1
    s = b2 - b1
    denom = cross_2d(r, s)
    if abs(denom) < 1e-8:
        return False
    qp = b1 - a1
    t = cross_2d(qp, s) / denom
    u = cross_2d(qp, r) / denom
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


def ray_segment_intersection(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
) -> float | None:
    def cross_2d(u: np.ndarray, v: np.ndarray) -> float:
        return float(u[0] * v[1] - u[1] * v[0])

    segment = seg_b - seg_a
    denom = cross_2d(ray_direction, segment)
    if abs(denom) < 1e-8:
        return None
    origin_delta = seg_a - ray_origin
    distance = cross_2d(origin_delta, segment) / denom
    segment_ratio = cross_2d(origin_delta, ray_direction) / denom
    if distance < 0.0 or not (0.0 <= segment_ratio <= 1.0):
        return None
    return float(distance)
