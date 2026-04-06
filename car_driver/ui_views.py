from __future__ import annotations

import math
import time
import tkinter as tk
from typing import Any

import numpy as np

from car_driver.ui_common import lerp_color


def _flatten_points(points: list[tuple[float, float]]) -> list[float]:
    flattened: list[float] = []
    for x, y in points:
        flattened.extend([x, y])
    return flattened


class GameCanvas(tk.Canvas):
    def __init__(self, master) -> None:
        super().__init__(master, highlightthickness=0, bg="#0a1720")
        self.snapshot: dict[str, Any] | None = None
        self.training_active = False
        self.overlay_text = "Evaluating the active brain."
        self.brain_source = "best"
        self.last_outcome_text = "Last eval: awaiting result"
        self.last_outcome_detail = (
            "The active brain will keep driving while training is paused."
        )
        self.last_outcome_kind = "neutral"
        self.render_interval_ms = 66
        self._redraw_after_id: str | None = None
        self._last_redraw_at = 0.0
        self.bind("<Configure>", lambda _event: self.request_redraw(force=True))

    def set_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.snapshot = snapshot
        self.request_redraw()

    def set_training_mode(self, active: bool, overlay_text: str | None = None) -> None:
        self.training_active = active
        if overlay_text is not None:
            self.overlay_text = overlay_text
        self.request_redraw(force=True)

    def set_brain_source(self, source: str) -> None:
        self.brain_source = source
        self.request_redraw()

    def set_evaluation_outcome(
        self,
        text: str,
        detail: str,
        kind: str = "neutral",
    ) -> None:
        self.last_outcome_text = text
        self.last_outcome_detail = detail
        self.last_outcome_kind = kind
        self.request_redraw()

    def request_redraw(self, force: bool = False) -> None:
        if not self.winfo_exists():
            return
        if self._redraw_after_id is not None:
            if not force:
                return
            self.after_cancel(self._redraw_after_id)
            self._redraw_after_id = None

        delay_ms = 0
        if not force:
            elapsed_ms = int((time.perf_counter() - self._last_redraw_at) * 1000.0)
            delay_ms = max(0, self.render_interval_ms - elapsed_ms)
        self._redraw_after_id = self.after(delay_ms, self._perform_redraw)

    def _perform_redraw(self) -> None:
        self._redraw_after_id = None
        self._last_redraw_at = time.perf_counter()
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 10)
        height = max(self.winfo_height(), 10)
        self._draw_background(width, height)

        if not self.snapshot or not self.snapshot.get("state") or not self.snapshot.get("track"):
            self.create_text(
                width / 2,
                height / 2,
                text="No active episode yet.\nStart training or let evaluation play.",
                fill="#f7fbff",
                font=("Segoe UI", 16, "bold"),
                justify="center",
            )
            return

        self._draw_first_person_scene(width, height)
        self._draw_top_down_inset(width, height)
        self._draw_status_badge(width)

        if self.training_active:
            self.create_rectangle(
                0,
                0,
                width,
                height,
                fill="#051017",
                stipple="gray50",
                outline="",
            )
            self.create_text(
                width / 2,
                height / 2,
                text=self.overlay_text,
                fill="#f8fcff",
                font=("Segoe UI", 18, "bold"),
                justify="center",
            )

    def _draw_background(self, width: int, height: int) -> None:
        horizon = height * 0.36
        steps = 10
        for index in range(steps):
            t = index / max(steps - 1, 1)
            color = (
                lerp_color("#092034", "#214f71", t / 0.55)
                if t < 0.55
                else lerp_color("#214f71", "#f0b36d", (t - 0.55) / 0.45)
            )
            y0 = horizon * index / steps
            y1 = horizon * (index + 1) / steps
            self.create_rectangle(0, y0, width, y1, fill=color, outline=color)
        self.create_oval(
            width * 0.70 - 40,
            horizon * 0.20 - 40,
            width * 0.70 + 40,
            horizon * 0.20 + 40,
            fill="#ffd89a",
            outline="",
        )
        self.create_rectangle(
            0,
            horizon,
            width,
            height,
            fill="#20442a",
            outline="",
        )

    def _draw_first_person_scene(self, width: int, height: int) -> None:
        assert self.snapshot is not None
        track = self.snapshot["track"]
        state = self.snapshot["state"]
        car_pos = np.array([state["x"], state["y"]], dtype=np.float32)
        heading = float(state["heading"])
        nearest_index = int(state["nearest_index"])
        forward = np.array([math.sin(heading), math.cos(heading)], dtype=np.float32)
        right = np.array([math.cos(heading), -math.sin(heading)], dtype=np.float32)

        centerline = track["centerline"]
        left_boundary = track["left_boundary"]
        right_boundary = track["right_boundary"]
        point_spacing = float(track["point_spacing"])
        road_width = float(track["road_width"])
        max_ahead_distance = 100.0
        stride = max(1, int(round(6.0 / max(point_spacing, 1e-6))))
        upper_horizon = height * 0.31
        road_depth_span = height * 0.56

        samples: list[
            tuple[tuple[float, float], tuple[float, float], tuple[float, float], float, float]
        ] = []
        for index in range(nearest_index, centerline.shape[0], stride):
            delta_center = centerline[index] - car_pos
            local_z = float(np.dot(delta_center, forward))
            if local_z <= 0.8:
                continue
            if local_z > max_ahead_distance:
                break
            local_center_x = float(np.dot(delta_center, right))
            delta_left = left_boundary[index] - car_pos
            delta_right = right_boundary[index] - car_pos
            local_left_x = float(np.dot(delta_left, right))
            local_right_x = float(np.dot(delta_right, right))
            local_left_z = float(np.dot(delta_left, forward))
            local_right_z = float(np.dot(delta_right, forward))
            if local_left_z <= 0.8 or local_right_z <= 0.8:
                continue

            depth_ratio = min(local_z / max_ahead_distance, 1.0)
            screen_y = upper_horizon + road_depth_span * (1.0 - depth_ratio ** 0.64)
            meter_scale = (
                width
                * (0.062 + 0.88 * ((1.0 - depth_ratio) ** 1.45))
                / max(road_width, 1e-6)
            )
            projected_left = (
                width * 0.5 + local_left_x * meter_scale,
                screen_y,
            )
            projected_right = (
                width * 0.5 + local_right_x * meter_scale,
                screen_y,
            )
            projected_center = (
                width * 0.5 + local_center_x * meter_scale,
                screen_y,
            )
            samples.append(
                (
                    projected_left,
                    projected_right,
                    projected_center,
                    local_z,
                    depth_ratio,
                )
            )

        if len(samples) < 2:
            return

        left_points = [sample[0] for sample in samples]
        right_points = [sample[1] for sample in samples]
        center_points = [sample[2] for sample in samples]

        road_polygon = _flatten_points(left_points + list(reversed(right_points)))
        self.create_polygon(
            *road_polygon,
            fill="#31353a",
            outline="",
        )

        self.create_line(
            *_flatten_points(left_points),
            fill="#f3ead4",
            width=2,
            smooth=True,
        )
        self.create_line(
            *_flatten_points(right_points),
            fill="#f3ead4",
            width=2,
            smooth=True,
        )

        center_dash_points = [
            center_points[index]
            for index in range(0, len(center_points), 2)
        ]
        if len(center_dash_points) >= 2:
            self.create_line(
                *_flatten_points(center_dash_points),
                fill="#ffd84f",
                width=2,
                dash=(10, 8),
                smooth=True,
            )

        for index in range(0, len(samples), 4):
            left_point, right_point, _, _, depth_ratio = samples[index]
            shoulder_tint = lerp_color("#d8f3ff", "#86aebd", depth_ratio)
            post_height = max(4.0, 14.0 * (1.0 - depth_ratio))
            post_offset = 5.0 + 8.0 * (1.0 - depth_ratio)
            self.create_line(
                left_point[0],
                left_point[1],
                left_point[0] - post_offset,
                left_point[1] - post_height,
                fill=shoulder_tint,
                width=2,
            )
            self.create_line(
                right_point[0],
                right_point[1],
                right_point[0] + post_offset,
                right_point[1] - post_height,
                fill=shoulder_tint,
                width=2,
            )

        for horizon_index in range(5):
            y = upper_horizon + 22 + horizon_index * 22
            self.create_line(0, y, width, y, fill="#26422a", dash=(2, 12))

        self.create_polygon(
            0,
            height,
            width * 0.20,
            height * 0.78,
            width * 0.80,
            height * 0.78,
            width,
            height,
            fill="#20252c",
            outline="",
        )

    def _draw_top_down_inset(self, width: int, height: int) -> None:
        assert self.snapshot is not None
        track = self.snapshot["track"]
        state = self.snapshot["state"]
        box = (width - 262, height - 232, width - 18, height - 18)
        self.create_rectangle(*box, fill="#09111c", outline="#264d7f", width=2)
        self.create_text(
            box[0] + 12,
            box[1] + 14,
            text="Debug Inset",
            fill="#74f7ff",
            font=("Consolas", 10, "bold"),
            anchor="w",
        )

        car_pos = np.array([state["x"], state["y"]], dtype=np.float32)
        heading = float(state["heading"])
        forward = np.array([math.sin(heading), math.cos(heading)], dtype=np.float32)
        right = np.array([math.cos(heading), -math.sin(heading)], dtype=np.float32)
        center = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.56)
        scale = 1.8

        def world_to_inset(point: np.ndarray) -> tuple[float, float]:
            delta = point - car_pos
            local_x = float(np.dot(delta, right))
            local_z = float(np.dot(delta, forward))
            return (center[0] + local_x * scale, center[1] - local_z * scale)

        nearest_index = int(state["nearest_index"])
        inset_range_m = 100.0
        inset_points = max(12, int(round(inset_range_m / max(float(track["point_spacing"]), 1e-6))))
        start = max(0, nearest_index - 20)
        stop = min(track["centerline"].shape[0], nearest_index + inset_points)
        for boundary, color, width_px in (
            (track["left_boundary"][start:stop:3], "#dfefff", 2),
            (track["right_boundary"][start:stop:3], "#dfefff", 2),
            (track["centerline"][start:stop:5], "#ffd84f", 1),
        ):
            inset_points = [world_to_inset(point) for point in boundary]
            if len(inset_points) >= 2:
                self.create_line(
                    *_flatten_points(inset_points),
                    fill=color,
                    width=width_px,
                    smooth=True,
                )

        trail = np.asarray(state["trail"], dtype=np.float32)[-48::4]
        trail_points = [world_to_inset(point) for point in trail]
        if len(trail_points) >= 2:
            self.create_line(
                *_flatten_points(trail_points),
                fill="#6ef7ff",
                width=1,
                smooth=True,
            )

        lidar_hits = np.asarray(state["lidar_hits"], dtype=np.float32)
        for hit in lidar_hits[::2]:
            x1, y1 = center
            x2, y2 = world_to_inset(hit)
            self.create_line(x1, y1, x2, y2, fill="#ff74f7", dash=(4, 3))

        car_length = 18
        car_width = 10
        car_points = [
            (-car_width * 0.5, car_length * 0.5),
            (car_width * 0.5, car_length * 0.5),
            (car_width * 0.5, -car_length * 0.5),
            (0.0, -car_length * 0.8),
            (-car_width * 0.5, -car_length * 0.5),
        ]
        polygon: list[float] = []
        for local_x, local_z in car_points:
            polygon.extend([center[0] + local_x, center[1] - local_z])
        self.create_polygon(polygon, fill="#ffb85e", outline="#fff3d8", width=2)

    def _draw_status_badge(self, width: int) -> None:
        badge_left = width - 250
        badge_top = 14
        badge_right = width - 16
        badge_bottom = 44
        self.create_rectangle(
            badge_left,
            badge_top,
            badge_right,
            badge_bottom,
            fill="#faf6ea",
            outline="",
        )
        self.create_text(
            (badge_left + badge_right) * 0.5,
            (badge_top + badge_bottom) * 0.5,
            text=(
                f"{'TRAINING' if self.training_active else 'EVALUATION'}"
                f" - {self.brain_source.upper()}"
            ),
            fill="#17303e",
            font=("Segoe UI", 10, "bold"),
        )


class GraphCanvas(tk.Canvas):
    def __init__(self, master) -> None:
        super().__init__(master, highlightthickness=0, bg="#08101f")
        self.history: list[Any] = []
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_history(self, history: list[Any]) -> None:
        self.history = list(history)
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 10)
        height = max(self.winfo_height(), 10)
        for index in range(24):
            color = lerp_color("#08101f", "#101c34", index / 23)
            self.create_rectangle(
                0,
                height * index / 24,
                width,
                height * (index + 1) / 24,
                fill=color,
                outline=color,
            )
        self.create_text(
            20,
            20,
            text="Training Progress",
            fill="#6ff7ff",
            font=("Consolas", 12, "bold"),
            anchor="w",
        )

        if not self.history:
            self.create_text(
                width / 2,
                height / 2,
                text=(
                    "Finish rate, mean distance, and score trends "
                    "appear here after training starts."
                ),
                fill="#87a2cf",
                font=("Segoe UI", 11),
                width=width * 0.8,
                justify="center",
            )
            return

        upper = (54, 58, width - 32, height * 0.40)
        lower = (54, height * 0.54, width - 32, height - 36)
        self._draw_plot_box(*upper)
        self._draw_plot_box(*lower)
        self._draw_grid(*upper)
        self._draw_grid(*lower)

        finish_rates = [float(item.finish_rate) for item in self.history]
        distance_ratios = [
            float(item.mean_distance) / max(float(getattr(item, "course_length", 1.0)), 1.0)
            for item in self.history
        ]
        best_scores = [float(item.best_score) for item in self.history]
        mean_scores = [float(item.mean_score) for item in self.history]
        finish_color = "#50f6ff"
        distance_color = "#a8ff6b"
        best_score_color = "#ffab4f"
        mean_score_color = "#ff74f7"
        self._draw_series(upper, finish_rates, 0.0, 1.0, finish_color)
        self._draw_series(upper, distance_ratios, 0.0, 1.0, distance_color)

        score_min = min(best_scores + mean_scores + [0.0])
        score_max = max(best_scores + mean_scores + [1.0])
        if math.isclose(score_min, score_max):
            score_max = score_min + 1.0
        pad = (score_max - score_min) * 0.08
        self._draw_series(lower, best_scores, score_min - pad, score_max + pad, best_score_color)
        self._draw_series(lower, mean_scores, score_min - pad, score_max + pad, mean_score_color)

        self.create_text(18, upper[1], text="100%", fill="#8db3d1", font=("Segoe UI", 9), anchor="w")
        self.create_text(18, upper[3] - 2, text="0%", fill="#8db3d1", font=("Segoe UI", 9), anchor="sw")
        self.create_text(18, lower[1], text=f"{score_max + pad:.0f}", fill="#8db3d1", font=("Segoe UI", 9), anchor="w")
        self.create_text(18, lower[3] - 2, text=f"{score_min - pad:.0f}", fill="#8db3d1", font=("Segoe UI", 9), anchor="sw")
        self.create_text(20, upper[1] - 14, text="Finish rate and mean progress", fill="#72f7ff", font=("Consolas", 10, "bold"), anchor="w")
        self.create_text(20, lower[1] - 14, text="Generation best and mean score", fill="#72f7ff", font=("Consolas", 10, "bold"), anchor="w")

        legend = [
            ("Finish", finish_color, f"{finish_rates[-1] * 100:.1f}%"),
            ("Progress", distance_color, f"{distance_ratios[-1] * 100:.1f}%"),
            ("Best", best_score_color, f"{best_scores[-1]:.1f}"),
            ("Mean", mean_score_color, f"{mean_scores[-1]:.1f}"),
        ]
        legend_x0 = max(240, width - 460)
        legend_x1 = width - 16
        self.create_rectangle(
            legend_x0,
            10,
            legend_x1,
            40,
            fill="#0b1630",
            outline="#244c88",
        )
        inner_width = max(legend_x1 - legend_x0 - 24, 160)
        column_width = inner_width / len(legend)
        for index, (label, color, value) in enumerate(legend):
            x = legend_x0 + 12 + index * column_width
            y = 25
            self.create_line(x, y, x + 16, y, fill=color, width=3)
            self.create_text(
                x + 22,
                y,
                text=f"{label} {value}",
                fill="#cfe7ff",
                font=("Segoe UI", 9),
                anchor="w",
            )

    def _draw_plot_box(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.create_rectangle(x0, y0, x1, y1, fill="#081121", outline="#244c88")

    def _draw_grid(self, x0: float, y0: float, x1: float, y1: float) -> None:
        for index in range(1, 4):
            y = y0 + (y1 - y0) * index / 4
            self.create_line(x0, y, x1, y, fill="#16365e", dash=(4, 3))

    def _draw_series(
        self,
        rect,
        values: list[float],
        minimum: float,
        maximum: float,
        color: str,
    ) -> None:
        x0, y0, x1, y1 = rect
        if len(values) == 1:
            values = [values[0], values[0]]
        points = []
        for index, value in enumerate(values):
            x = x0 + (x1 - x0) * index / max(len(values) - 1, 1)
            norm = (value - minimum) / max(maximum - minimum, 1e-6)
            y = y1 - norm * (y1 - y0)
            points.extend([x, y])
        self.create_line(*points, fill=color, width=3, smooth=True)
        for index in range(0, len(points), 2):
            self.create_oval(
                points[index] - 3,
                points[index + 1] - 3,
                points[index] + 3,
                points[index + 1] + 3,
                fill=color,
                outline="",
            )


class NetworkCanvas(tk.Canvas):
    def __init__(self, master) -> None:
        super().__init__(master, highlightthickness=0, bg="#0b1723")
        self.visual_data: dict[str, Any] | None = None
        self.title = "Active Brain"
        self._visual_signature: Any = None
        self._layout_cache_key: Any = None
        self._layout_cache: dict[str, Any] = {}
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_visualization(self, visual_data: dict[str, Any] | None, title: str) -> None:
        signature = self._build_visual_signature(visual_data)
        if signature == self._visual_signature and title == self.title:
            return
        self.visual_data = visual_data
        self.title = title
        self._visual_signature = signature
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 10)
        height = max(self.winfo_height(), 10)
        for index in range(26):
            color = lerp_color("#060d18", "#101d35", index / 25)
            self.create_rectangle(
                0,
                height * index / 26,
                width,
                height * (index + 1) / 26,
                fill=color,
                outline=color,
            )
        self.create_text(
            18,
            22,
            text=self.title,
            fill="#72f7ff",
            font=("Consolas", 12, "bold"),
            anchor="w",
        )

        if not self.visual_data:
            self.create_text(
                width / 2,
                height / 2,
                text="The actor network will appear here once a brain is available.",
                fill="#8aa5c8",
                font=("Segoe UI", 11),
                width=width * 0.8,
                justify="center",
            )
            return

        layer_sizes = list(self.visual_data["layer_sizes"])
        weight_matrices = list(self.visual_data["weights"])
        activations = list(self.visual_data["activations"])
        input_names = list(self.visual_data.get("input_names", []))
        output_names = list(self.visual_data.get("output_names", []))
        log_std = list(self.visual_data.get("log_std", []))

        layout = self._get_layout(width, height, layer_sizes)
        layers = layout["layers"]
        x_positions = layout["x_positions"]
        left = layout["left"]
        right = layout["right"]
        max_abs = 1e-6
        for matrix in weight_matrices:
            max_abs = max(max_abs, abs(matrix).max())

        for layer_index, matrix in enumerate(weight_matrices):
            for out_idx, right_point in enumerate(layers[layer_index + 1]):
                for in_idx, left_point in enumerate(layers[layer_index]):
                    weight = float(matrix[out_idx][in_idx])
                    norm = abs(weight) / max_abs
                    color = "#4af7ff" if weight >= 0 else "#ff74f7"
                    width_px = 0.6 + 2.8 * (norm ** 1.3)
                    kwargs = {
                        "fill": color,
                        "width": width_px,
                        "smooth": True,
                        "splinesteps": 12,
                    }
                    if norm <= 0.35:
                        kwargs["stipple"] = "gray50"
                    self.create_line(
                        left_point[0],
                        left_point[1],
                        right_point[0],
                        right_point[1],
                        **kwargs,
                    )

        radius = layout["radius"]
        for layer_index, layer_points in enumerate(layers):
            for node_index, (x, y) in enumerate(layer_points):
                glow = radius + 3
                glow_fill = (
                    "#123d4d"
                    if layer_index == 0
                    else "#4b214d"
                    if layer_index == len(layers) - 1
                    else "#18355d"
                )
                self.create_oval(
                    x - glow,
                    y - glow,
                    x + glow,
                    y + glow,
                    fill=glow_fill,
                    outline="",
                )
                fill = (
                    "#2ae8ff"
                    if layer_index == 0
                    else "#ff74f7"
                    if layer_index == len(layers) - 1
                    else "#d7edff"
                )
                self.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=fill,
                    outline="#f3fbff",
                )
                if layer_index == 0 and node_index < len(input_names):
                    self.create_text(
                        left - 12,
                        y,
                        text=input_names[node_index],
                        fill="#cce4ff",
                        font=("Segoe UI", 8),
                        anchor="e",
                    )
                if layer_index == len(layers) - 1 and node_index < len(output_names):
                    self.create_text(
                        right + 12,
                        y,
                        text=output_names[node_index],
                        fill="#ffd3fb",
                        font=("Segoe UI", 8),
                        anchor="w",
                    )

        for index, activation in enumerate(activations):
            if index + 1 >= len(x_positions):
                continue
            x = (x_positions[index] + x_positions[index + 1]) / 2
            self.create_rectangle(
                x - 32,
                34,
                x + 32,
                56,
                fill="#0f1c34",
                outline="#2eeeff",
            )
            self.create_text(
                x,
                45,
                text=activation,
                fill="#e8fbff",
                font=("Consolas", 9, "bold"),
            )

        topology = " -> ".join(str(size) for size in layer_sizes)
        self.create_text(
            18,
            height - 38,
            text=f"Topology: {topology}",
            fill="#b9d7ff",
            font=("Consolas", 9),
            anchor="w",
        )
        if log_std:
            log_std_text = ", ".join(f"{value:.2f}" for value in log_std[:4])
            self.create_text(
                18,
                height - 18,
                text=f"Policy log std: {log_std_text}",
                fill="#b9d7ff",
                font=("Consolas", 9),
                anchor="w",
            )

    def _build_visual_signature(
        self,
        visual_data: dict[str, Any] | None,
    ) -> Any:
        if visual_data is None:
            return None
        weights = visual_data.get("weights", [])
        return (
            tuple(visual_data.get("layer_sizes", [])),
            tuple(matrix.shape for matrix in weights),
            tuple(hash(matrix.tobytes()) for matrix in weights),
            tuple(visual_data.get("activations", [])),
            tuple(visual_data.get("log_std", [])),
        )

    def _get_layout(
        self,
        width: int,
        height: int,
        layer_sizes: list[int],
    ) -> dict[str, Any]:
        key = (width, height, tuple(layer_sizes))
        if key == self._layout_cache_key:
            return self._layout_cache

        left = 92
        top = 84
        right = width - 92
        bottom = height - 72
        x_positions = [
            left + (right - left) * index / max(len(layer_sizes) - 1, 1)
            for index in range(len(layer_sizes))
        ]
        max_nodes = max(layer_sizes)

        def positions(count: int, x: float) -> list[tuple[float, float]]:
            if count == 1:
                return [(x, (top + bottom) / 2)]
            spacing = (bottom - top) / max(count - 1, 1)
            return [(x, top + idx * spacing) for idx in range(count)]

        layers = [
            positions(size, x_positions[index])
            for index, size in enumerate(layer_sizes)
        ]
        radius = max(4, min(10, int((bottom - top) / max_nodes * 0.24)))
        self._layout_cache_key = key
        self._layout_cache = {
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "x_positions": x_positions,
            "layers": layers,
            "radius": radius,
        }
        return self._layout_cache
