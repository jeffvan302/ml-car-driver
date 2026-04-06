# Project Requirements

## Purpose

Create a Python car driving game and Deep Reinforcement Learning training tool with a desktop GUI, a self-implemented 2D driving environment rendered as a first-person experience, and a PyTorch PPO training system.

The project should intentionally mirror the overall structure and workflow of the reference rocket project while replacing the rocket physics and rewards with a solo timed-run car driving game.

## Functional Requirements

1. The project must launch from `run.py`.

2. `run.py` must support these entry modes:
   * `python run.py` for the desktop GUI
   * `python run.py gui`
   * `python run.py headless-train --generations N --games M`
   * `python run.py headless-train --load checkpoint.pt --generations N --games M`
   * `python run.py smoke-test`

3. All installable third-party dependencies must be listed in `requirements.txt`, and `pip install -r requirements.txt` must work on Python 3.12.

4. Python 3.12 is the target runtime. PyTorch must be used for the PPO actor-critic model and training loop.

5. The GUI must use a Python-native desktop toolkit that runs reliably in the target Python 3.12 environment. Current implementation target: `tkinter`.

6. The codebase must remain modular:
   * game/environment logic separated from GUI
   * PPO/model/training logic separated from GUI
   * UI controls separated from UI drawing/rendering
   * config validation separated from GUI widgets

7. Core modules must be independently testable from the command line:
   * `environment.py`
   * `ppo.py`
   * `training.py`

8. The driving environment must be self-implemented and must not depend on Gym or another prebuilt driving simulator.

9. The game must model a solo timed run on a road with no traffic and no moving obstacles. The challenge is road-following speed and turn handling.

10. The simulation must use a simple 2D car model under the hood and render the gameplay as a first-person view from behind the steering wheel looking through the windshield.

11. The default implementation target for the car physics should be a simple kinematic bicycle model or an equivalent single-track approximation rather than a full high-fidelity vehicle simulator.

12. The environment must expose a four-output action space corresponding to:
   * gas
   * brake
   * left
   * right

13. The actor must output continuous control values, and the environment must convert those values into effective controls using a dead-zone threshold.

14. The default control dead-zone threshold must be `0.1`. Any control intensity below `0.1` must be treated as inactive.

15. Action resolution must obey these rules:
   * `left` and `right` both active means steer straight
   * `gas` and `brake` both active means brake wins
   * if neither `gas` nor `brake` is active, the car must coast and gradually slow down
   * reverse driving is not allowed

16. PPO must use an actor-critic design:
   * actor outputs driving control actions
   * critic outputs scalar value estimates for training
   * critic output is not required in the GUI

17. PPO must use batched multi-environment rollout collection rather than only one episode at a time.

18. Observation normalization must be included in training and must also be reused during evaluation and playback.

19. Training stop behavior must be precise:
   * `Pause` finishes the current generation
   * `Stop` finishes currently active episodes, does not start new ones, and may produce a partial generation

20. Partial generations must not overwrite the saved "best brain so far."

21. The app must track both:
   * current brain
   * best brain so far

22. "Best brain so far" must be determined primarily by finish rate and secondarily by mean score.

23. The application must support resuming training from a saved session checkpoint.

24. A resumed session must restore at least:
   * current brain
   * best brain so far
   * optimizer state when available
   * current observation normalization state
   * best observation normalization state
   * training history
   * best metrics
   * saved GUI configuration

25. Resuming training must work even after the user changes physics and reward settings, as long as the network architecture remains compatible.

26. If physics or rewards are changed after loading a session, training must continue from the loaded current brain using the updated physics/reward configuration.

27. During pause or after training, the app must continuously evaluate the selected brain in the live game view.

28. The GUI layout must contain three major regions:
   * left scrollable configuration panel
   * center panel split vertically into game view and training graph
   * right resizable neural-network visualization panel

29. The left panel must scroll vertically only and contain:
   * session controls
   * brain controls
   * PPO hyperparameters
   * network layer configuration
   * car physics settings
   * track generation settings
   * reward and penalty settings

30. Session controls must include:
   * start/resume training
   * pause after current generation
   * stop after active episodes finish

31. Brain/session controls must include:
   * toggle between current brain and best brain so far
   * save session
   * load session

32. The left panel must allow editing hidden layer count, hidden layer size, and hidden layer activation per layer.

33. Default network settings must be:
   * hidden layer 1: `8`, `relu`
   * hidden layer 2: `40`, `relu`
   * hidden layer 3: `8`, `relu`
   * actor output activation: `tanh`

34. The left panel must allow editing at least these PPO values:
   * target generations
   * games per generation
   * parallel environment count
   * learning rate
   * gamma
   * gae lambda
   * clip range
   * entropy coefficient
   * value coefficient
   * PPO epochs
   * minibatch size
   * max grad norm
   * action std or equivalent exploration control
   * seed

35. Default games per generation must be `25`.

36. The left panel must allow editing at least these driving physics values:
   * max speed
   * acceleration
   * brake deceleration
   * coasting deceleration
   * steering rate or equivalent turn response
   * max steering angle or equivalent steering limit
   * car length
   * car width
   * dt
   * max steps
   * stalled timeout seconds

37. The default stalled timeout must be `10` seconds.

38. The physics section must also include fuel settings:
   * fuel tank volume in liters
   * initial fuel
   * fuel usage in liters per second at gas use

39. Fuel consumption must scale with gas control intensity above the dead zone. When the fuel tank is empty, gas input must no longer produce acceleration.

40. Forward speed must remain clamped to the valid range from `0` to `max_speed`. Braking and coasting must not cause negative speed.

41. The left panel must allow editing at least these sensor settings:
   * lidar range
   * lidar beam count
   * lidar angle spacing

42. Default sensor settings must be:
   * lidar range: `50 m`
   * lidar beam count: `13`
   * one forward beam plus six beams on each side
   * `10` degree spacing between adjacent beams
   * beam coverage from `-60` degrees to `+60` degrees relative to forward

43. The lidar system must detect road edges or any other blocking boundary in the beam path. Since the game is a solo timed run, the initial version does not require traffic or obstacle actors.

44. Lidar observations must be normalized to `0.0..1.0`, where:
   * `1.0` means no hit within the configured lidar range
   * a nearer hit produces a smaller normalized value

45. The left panel must allow editing at least these track generation values:
   * target course length
   * road width
   * minimum straight length
   * maximum straight length
   * minimum turn radius
   * maximum turn radius
   * minimum turn angle
   * maximum turn angle
   * random seed override or generation seed

46. Default target course length must be `3000 m`.

47. The track generator must build roads from straight, left-turn, and right-turn segments.

48. Turn segments must support variable radius and variable arc length rather than forcing every turn to use the same curvature.

49. Track generation must enforce constraints that keep roads drivable, including:
   * valid min/max ranges
   * no zero-width road
   * no impossible immediate geometry transitions
   * no turns tighter than the configured minimum radius
   * no generated segments that self-intersect within a single episode unless explicitly allowed in a future mode

50. The environment must expose an observation vector that includes:
   * normalized speed
   * 13 normalized lidar beam values
   * heading error relative to road direction
   * lateral offset from road center

51. The default observation vector size must therefore be `16`.

52. Speed must be normalized by the configured max speed.

53. The episode must begin with the car on the generated road near the start of the course, aligned closely enough to make the run solvable while still allowing some randomized variation. The spawned car body must begin fully inside the drivable road bounds.

54. Episode termination conditions must include at minimum:
   * finish
   * crash or off-road loss
   * timeout by max steps
   * stalled state after `10` seconds of insufficient movement

55. A crash or off-road loss must occur when any part of the car body leaves the drivable road bounds.

56. The environment must define a score or reporting summary that reflects both distance reached and elapsed time.

57. The left panel must allow editing at least these rewards and penalties:
   * finish bonus
   * progress reward scale
   * centerline bonus
   * heading alignment bonus
   * alive bonus
   * step penalty
   * steering penalty
   * brake penalty
   * crash penalty
   * off-road penalty
   * timeout penalty
   * stall penalty
   * optional fuel-efficiency bonus or penalty

58. Reward shaping should encourage fast and controlled driving by combining:
   * positive reward for forward progress
   * positive reward for staying near the road center
   * positive reward for pointing in the road direction
   * small penalty per step
   * large negative reward for crashing or leaving the road
   * negative reward for stalling
   * strong positive reward for reaching the finish

59. The GUI must validate settings before training begins, including cross-field checks such as:
   * positive `max_speed`, acceleration, brake deceleration, coasting deceleration, and `dt`
   * positive road width and course length
   * valid turn and straight segment ranges
   * `min <= max` for all ranged generator settings
   * positive lidar range
   * supported activation names
   * non-negative fuel values
   * stalled timeout greater than `0`

60. Validation must distinguish between blocking errors and non-blocking warnings.

61. The game view must clearly visualize:
   * the first-person road scene
   * road boundaries
   * horizon or visual depth cues sufficient for steering interpretation
   * telemetry overlay
   * evaluation/training state
   * active speed
   * fuel remaining
   * current episode outcome when available

62. The center panel must also include a small top-down debug inset that shows:
   * road layout near the car
   * car footprint and heading
   * lidar rays
   * nearby road boundaries

63. During training, the game animation must pause to reduce rendering overhead, while the graph and network panel update after each generation.

64. The graph view must show:
   * finish rate
   * generation best score
   * generation mean score
   * generation mean distance or equivalent progress metric

65. The right panel must render the actor network with all nodes shown.

66. Connections in the network view must visually emphasize stronger weights more than weaker weights.

67. The network view should be efficient enough to handle larger user-defined networks by caching layout and avoiding unnecessary redraw work.

68. The neural-network visualization must update when:
   * a new generation completes
   * the selected brain changes
   * the loaded checkpoint changes
   * the canvas size changes

69. The left panel must include an `Apply Physics` or equivalent action that updates the live evaluation environment without restarting the whole app.

70. Save/load must persist:
   * current brain weights
   * best brain weights
   * optimizer state when available
   * config/settings from the left panel
   * training history
   * best metrics
   * current observation normalization state
   * best observation normalization state
   * checkpoint metadata

71. Checkpoint metadata should include at least:
   * schema version
   * save timestamp
   * generation count
   * total episodes
   * total steps
   * best metrics
   * last generation summary
   * best generation summary
   * whether the checkpoint is resume-capable

72. Loading an older checkpoint format should remain backward compatible when possible by falling back to a single loaded brain if separate current/best session data is unavailable.

73. The project must include automated tests and smoke checks that verify:
   * environment observation shape and step behavior
   * lidar normalization and hit behavior
   * track generation respects configured ranges
   * trainer can complete a short run
   * checkpoint round-trip works
   * config validation catches bad input
   * resumed training from a saved session works
   * stalled episodes terminate correctly
   * fuel depletion disables gas acceleration

74. Minimum validation commands for project completion:
   * `python run.py smoke-test`
   * `python -m unittest discover -s tests`
   * `python run.py headless-train --generations 1 --games 4`
   * `python run.py headless-train --load checkpoint.pt --generations 1 --games 2`

75. During pause-state or post-training evaluation playback, the GUI must persistently display the most recent terminal evaluation outcome until the next evaluation episode finishes.

76. The evaluation outcome display must distinguish at minimum:
   * successful finish
   * crash or off-road loss
   * timeout
   * stall

77. The GUI should also maintain visible running counts for these evaluation outcomes during the current evaluation watch session so the user can quickly judge whether the observed brain is improving.

78. The initial implementation target package structure should mirror the reference project closely, using equivalent modules such as:
   * `car_driver/environment.py`
   * `car_driver/ppo.py`
   * `car_driver/training.py`
   * `car_driver/validation.py`
   * `car_driver/ui_app.py`
   * `car_driver/ui_controls.py`
   * `car_driver/ui_views.py`
   * `car_driver/ui_common.py`
   * `car_driver/tk_gui.py`

79. The source of truth for implementation scope should be this document until a more detailed design or test specification supersedes it.
