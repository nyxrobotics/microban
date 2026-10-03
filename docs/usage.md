# Using Microban

This guide covers day-to-day operation once the robot has been set up (see the
[Deployment Guide](deployment.md)). You drive the robot from your computer with the
`Makefile`, which talks to the Pi over SSH. You can use the keyboard or a Bluetooth 
gamepad to control it, and optionally run it

> [!IMPORTANT]
> Always run `make shutdown` before cutting power to the robot. This is **not**
> automatic — powering off the Pi without a clean shutdown can corrupt the SD card.
> Wait 10-15 s after the command before flipping the power switch off, to give the
> Pi time to actually halt.

## Makefile commands

Run these from the repository root on your computer. They target the host `microban`
by default; add `HOST=microban-ext` to operate over the secondary network (see the
[Deployment Guide](deployment.md)).

| Command | What it does |
| :--- | :--- |
| `make run` | Sync the code and start the control loop on the robot (50 Hz). Stays attached to your terminal for live control. |
| `make stop` | Stop the control loop and disable torque on all motors. |
| `make shutdown` | Power off the Pi cleanly. |
| `make setup` | Sync the code and (re)install dependencies on the robot (`uv sync --frozen`). Run after changing dependencies. |
| `make sync` | Sync your local copy to the robot without touching dependencies. |
| `make imu` | Stream the robot's IMU/gyro readings to your terminal. |
| `make voltage` | Read the voltage of all motors. |
| `make voltage ID=<id>` | Read the voltage of motor `<id>`. |
| `make sim` | Run the MuJoCo simulation locally (no robot needed). |
| `make viewer` | Open the MuJoCo viewer locally (no robot needed). |
| `make teleop-run` | Run the robot with the external PICO/WebXR UDP input (see [Teleoperation](teleop.md)). |
| `make teleop-sim` | Run MuJoCo with the same UDP input, for testing before hardware. |
| `make camera-stream-enable` | Install/start the optional stereo MJPEG service after the camera is connected. |

## Running the robot

1. Place the robot on a stable surface, or hold it securely — on start it enables
   torque and ramps to its neutral pose.
2. `make run` — the control loop starts at 50 Hz and stays attached to your terminal. Some latency is expected due to the SSH connection.
3. Toggle moves and drive the robot (see below).
4. `make stop` (or press `q`) to stop; `make shutdown` to power off.

## Controlling with the keyboard

| Key | Action |
| :--- | :--- |
| `v` | toggle the **walk** move |
| `h` | toggle the **head** move |
| `s` | toggle the **squat** move |
| arrows | `vx` (up/down), `vtheta` (left/right) |
| `x` | reset velocity to zero |
| `i` | toggle the IMU/gyro display |
| `q` | stop the control loop |

## Controlling with a gamepad

A Bluetooth Xbox controller can be used instead of the keyboard. The detailed explanation of the gamepad usage is in [Gamepad Guide](gamepad.md). 

Using a gamepad allows to drive the robot through two different modes: with a terminal (SSH) or fully headless (no SSH, no terminal). The second mode is particularly useful for demonstration purposes, due to the fact that it allows to drive the robot without any computer connected to it.

## Moves

Moves are toggled independently and run on top of the neutral pose:

- **Walk** (`v` / gamepad **A**) — a reinforcement-learning policy. Once active, the
  velocity command drives it: `vx` (forward/back), `vy` (lateral), `vtheta` (turn),
  set from the arrow keys or the gamepad sticks.
- **Head** (`h`) — oscillates the head.
- **Squat** (`s`) — squat motion computed with inverse kinematics.

### Velocity command

Every input source emits a **normalized** command in `[-1, 1]` per axis; the scheduler
maps it to physical limits with `scale_velocity()`, so the behavior is identical for
keyboard, gamepad and sim. Defaults (in [constants.py](../src/constants.py)):

| Axis | Max |
| :--- | :--- |
| `vx` (forward) | +0.7 |
| `vx` (backward) | -0.5 |
| `vy` (lateral) | ±0.3 |
| `vtheta` (turning in place, `vx = vy = 0`) | ±3.0 |
| `vtheta` (while translating) | ±1.5 |

## Real-robot joint offsets

If one robot holds a joint slightly off from where the policies expect it (a servo
horn mounted slightly off, a slightly bent bracket, a robot that wants its ankles
pitched forward a little), correct it with the per-joint table
`HARDWARE_JOINT_OFFSET_DEG` in [constants.py](../src/constants.py). It lists all 21
joints and every value is `0.0` by default.

- Units: degrees of the **logical** joint coordinate (the one used by the policies,
  training, `NEUTRAL_POSE` and the observations; before `MOTOR_SIGN`). A positive value
  moves the real joint further in that joint's positive direction.
- Convention: `servo command = MOTOR_SIGN * (logical target + offset)` and
  `logical measurement = MOTOR_SIGN * servo reading - offset`.
- Scope: applied only inside `RobotController` (the real servo bus), on every goal
  write and every position read. Every move (walk, PICO tracking, get-up, the A /
  policy-off neutral pose, arms, head and neck) and every input source (keyboard,
  gamepad, GC300, PICO) gets it automatically, and the policies keep observing
  training coordinates. Velocities and currents are unchanged. MuJoCo / placo
  simulation (`make sim`, `make viewer`) and training ignore it.
- Measuring: with all offsets `0.0`, hold the joint at a known true angle (in the
  logical coordinate) and read its logical position `m` from the runtime (it already
  includes `MOTOR_SIGN`). Then `offset = m - true angle`, converted to degrees. Getting
  the sign wrong doubles the error instead of removing it.
- Limit: each value must be finite and at most `HARDWARE_JOINT_OFFSET_MAX_RAD`
  (0.2 rad, about 11.5 deg); otherwise the runtime refuses to start. Nonzero values are
  printed once when the runtime starts (`Hardware joint offsets (...)`). Offsets are
  calibration trims; they do not keep goals in range.
- Servo range: policies have no software clip. Every policy target is
  `clip(HOME + raw * 1.0, -pi, +pi)`, the servo's one-turn goal range. After sign and
  offset, `RobotController` saturates every servo goal into
  `[SERVO_GOAL_MIN_RAD, SERVO_GOAL_MAX_RAD] = [-pi, pi - 2*pi/4096]` rad, which is raw
  0..4095 in XC330 Position Control mode (rustypot: `raw = (rad + pi) * 4096 / (2*pi)`).
  An in-range goal is sent unchanged. A goal past an edge is sent as that edge, and the
  cached goal becomes the edge mapped back to the logical coordinate. At startup the
  runtime reads every servo's Operating Mode and Min/Max Position Limit: a servo that
  answers with a mode other than Position Control (3), or with fewer than two raw values
  between its limits, stops startup with an error naming the joints and every servo's
  torque OFF; a servo with narrower limits (read the same twice) gets them as its own
  goal range (intersected with 0..4095) and is printed once (`Servo position limits: ...
  raw [min, max] -> logical [lo, hi] rad`, plus a WARNING if its neutral pose is outside);
  a servo that does not answer keeps the full range with one warning line and is read
  again when it first answers a position read (a non-position mode found then stops the
  runtime the same way).

Example: to pitch both feet 1 degree forward (the old GC300-only ankle bias), set
`"left_ankle_pitch": -1.0` and `"right_ankle_pitch": -1.0`, then `make sync` and restart
the runtime.

## Developing: adding your own moves

Each behavior is a subclass of `Move` ([src/moves/move.py](../src/moves/move.py)) with
a simple lifecycle driven by the scheduler:

- `preload()` — optional, called once before the loop starts (load heavy resources).
- `on_start(obs, command)` — called each tick while *starting*; set
  `self.state = MoveState.ACTIVE` when ready (e.g. after ramping in).
- `step(obs, command)` — called each tick while *active*; write your target joint
  angles into `command.target_angles`.
- `on_stop(obs, command)` — called each tick while *stopping*; set
  `self.state = MoveState.INACTIVE` when done (e.g. after ramping back to neutral).

To add a move:

1. Create a new file in [src/moves/](../src/moves/) with a class subclassing `Move`.
   Use [rotate_head.py](../src/moves/rotate_head.py) (a simple oscillation) or
   [squat.py](../src/moves/squat.py) (inverse kinematics with placo) as a template.
2. Register it in [src/main.py](../src/main.py): add it to the `moves` dict passed to
   the `Scheduler`, and add a trigger — a key in `MOVE_KEYS` (keyboard) and/or a button
   in `GAMEPAD_BUTTON_MOVES` (gamepad).
3. In `step()`, read the robot state from `obs.robot_state` (motor positions and
   velocities, IMU gyro, projected gravity) and write your targets into
   `command.target_angles`.

### Training your own walk (or other RL) policies

The walk move runs an ONNX policy trained in simulation. 

PICO 4 Ultra network control and the left-trigger-held hybrid policy
are documented in [contract-v12 PICO policy runtime](pico_teleop_v12_runtime.md).
You can train your own walking — or other learned skills — and drop the resulting `.onnx` file into [src/agents/](../src/agents/) to use it on the robot. Check the repository [MarcDcls/mjlab_microban](https://github.com/MarcDcls/mjlab_microban) for the training pipeline. 

If you achieve some interesting results, don't hesitate to make a pull request to the repository as it is also a community-driven project!
