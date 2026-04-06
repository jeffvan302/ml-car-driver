# Car Driver Lab

Car Driver Lab is a Python 3.12 desktop application and PPO training tool for a self-implemented solo timed-run driving game.

It combines:

- a custom 2D car environment with procedural road generation
- a first-person driving view with a top-down debug inset
- a PyTorch actor-critic PPO trainer
- a desktop GUI for tuning physics, lidar, track generation, rewards, and network layers
- resumable checkpoint save/load support

For the tracked implementation scope, see [project_requirements.md](./project_requirements.md).

## Features

- `run.py` launches the GUI or headless training
- four continuous control outputs: `gas`, `brake`, `left`, `right`
- observation space built from normalized speed, lidar beams, heading error, and lateral offset
- random roads built from straight, left-turn, and right-turn segments
- fuel use, coasting, braking, stall detection, and finish/crash/timeout episode outcomes
- side-by-side game, graph, and network visualization panels
- current brain vs best brain evaluation switching

## Requirements

- Python 3.12
- `pip`
- packages from `requirements.txt`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running

Launch the GUI:

```bash
python run.py
```

Explicit GUI mode:

```bash
python run.py gui
```

Load a saved session directly into the GUI:

```bash
python run.py gui --load checkpoint.pt
```

Run headless training:

```bash
python run.py headless-train --generations 5 --games 8
```

Resume from a saved checkpoint:

```bash
python run.py headless-train --load checkpoint.pt --generations 3 --games 8
```

Run the built-in smoke test:

```bash
python run.py smoke-test
```

## GUI Layout

- Left panel: session controls, PPO settings, network layers, physics, sensors, track generation, and rewards
- Center panel: first-person driving view on top and training graph below
- Right panel: actor-network visualization

## Project Structure

- `run.py`: CLI entry point and GUI launcher
- `car_driver/environment.py`: custom driving environment, lidar, and track generation
- `car_driver/ppo.py`: actor-critic model and PPO action utilities
- `car_driver/training.py`: PPO training loop, checkpointing, resume support, smoke test helpers
- `car_driver/validation.py`: config validation rules
- `car_driver/ui_app.py`: main GUI application and training coordination
- `car_driver/ui_controls.py`: scrollable configuration controls
- `car_driver/ui_views.py`: game, graph, and network canvases
- `car_driver/ui_common.py`: shared UI helpers
- `car_driver/tk_gui.py`: GUI launcher wrapper
- `tests/test_smoke.py`: smoke and regression tests

## Validation

Recommended commands:

```bash
python -m unittest discover -s tests
python run.py smoke-test
python run.py headless-train --generations 1 --games 4
python run.py headless-train --load checkpoint.pt --generations 1 --games 2
```

## Notes

- The environment is self-implemented and does not depend on Gym or a third-party driving simulator.
- PPO uses batched multi-environment rollout collection.
- Observation normalization is shared between training and evaluation so resumed brains behave consistently.
