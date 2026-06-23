# gym-pybullet-drones — Tactical Behaviors & Swarm Extension

> **This is a fork of [learnsyslab/gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones)**
> by [Jacopo Panerati](https://github.com/JacopoPan) and the [Learning Systems and Robotics Lab](https://github.com/learnsyslab).
> Original work © 2020 Jacopo Panerati, MIT License. See [Citation](#citation) below.

This fork adds hand-coded **macro behaviors** and **multi-drone swarm coordination** on top of the base simulator, enabling scripted tactical missions without reinforcement learning.

---

## What's New in This Fork

### Behaviors — `gym_pybullet_drones/behaviors/`

A finite-state machine (FSM) manager that sequences four macro behaviors for a single drone:

| Behavior | Description |
|---|---|
| **Transit** | Point-to-point flight with trapezoidal velocity profile |
| **Recon** | Lawnmower or spiral search over a disk-shaped area |
| **Loiter** | Circular orbit tracking a (possibly moving) target |
| **Strike** | Terminal dash onto a target with deceleration |

### Swarm — `gym_pybullet_drones/swarm/`

A `SwarmCoordinator` that assigns one shared mission across N drones:

- **Task allocation** — splits Recon across drones with staggered search patterns (lawnmower or spiral); staggers Loiter phase angles
- **Deconfliction** — reactive separation filter keeps drones apart (relaxed during Strike)
- **Formation transit** — line-abreast formation for coordinated repositioning

---

## Behavior Implementation

All behaviors share the same interface: `reset(state, **params)` initializes the trajectory from the current drone state, and `step(state) → Setpoint` is called once per control step. They have no PyBullet/gym dependency — they consume only the 20-dim state vector and emit position/velocity setpoints for the low-level PID.

```
BehaviorManager (FSM)
├── mission queue: deque[(BehaviorType, params), ...]
├── step(state) → Setpoint
│     └── current behavior.is_done() → pop next → reset → step
└── each behavior is a reusable instance (pre-allocated in registry)
```

### Transit

Plans the full trapezoidal velocity profile at `reset` time. At each `step`, elapsed time `t` maps to an arc-length position along the straight line of sight.

```
long distance:  accelerate (t_acc) → cruise (t_cruise) → decelerate (t_acc)
short distance: triangle profile (no cruise phase)
```

`done` when: planned time elapsed **and** drone physically within `reach_tol` of the goal (prevents FSM from advancing while the drone still lags).

### Recon

Computes the entire coverage polyline at `reset` time and stores it as a sequence of waypoints with cumulative arc lengths. At each `step`, integrates `s += speed * dt` and interpolates position along the polyline.

Two coverage patterns (selected via `--recon_mode` in the swarm, `--pattern` in the single-drone demo):
- **lawnmower** — square (rectilinear) spiral expanding outward from the center with right-angle turns; same radial growth rate as the Archimedean spiral
- **spiral** — Archimedean spiral expanding outward from the center

`done` when: `s >= total_len` (polyline exhausted).

### Loiter

No pre-planning. At each `step`, computes the orbit angle `a = phase + ω*t` and emits the corresponding point on the circle. If the target is a callable, the orbit center follows it in real time.

```python
orbit_radius = standoff_alt / tan(depression_deg)  # derived from sensor FOV
omega        = orbit_speed / orbit_radius
pos          = center + [R*cos(a), R*sin(a), standoff_alt]
yaw          = towards center (sensor stares inward)
```

`done` when: `t >= duration`.

### Strike

No pre-planning. At each `step`, recomputes the vector from current position to target and pins the position setpoint to the target while pushing a strong velocity feed-forward. Speed is ramped down linearly within `decel_dist` to prevent overshoot.

```python
speed = dash_speed * min(1.0, dist / decel_dist)
Setpoint(pos=target, vel=speed * direction, relax_safety=True)
```

`done` when: `dist < hit_radius` (impact) or `timeout` exceeded.

### Summary

| | Transit | Recon | Loiter | Strike |
|---|---|---|---|---|
| Planning | at `reset` (full profile) | at `reset` (full polyline) | none (per-step) | none (per-step) |
| Moving target | no | no | yes (callable) | yes (callable) |
| `done` trigger | physical arrival | arc length exhausted | elapsed time | distance or timeout |

Transit and Recon are **open-loop** trajectory followers; Loiter and Strike are **closed-loop** target trackers.

---

## Swarm Behavior

In the swarm, a single macro command is expanded by `allocation.expand()` into one sub-task per drone. A **phase barrier** in `SwarmCoordinator` ensures every drone finishes the current phase before the swarm advances to the next — the slowest drone sets the pace.

### Transit (swarm)

One shared target is converted into N formation slots, then assigned to drones in a way that prevents path crossings.

**`mode="formation"` — line-abreast to a waypoint**

```
shared target: [1.2, 0.0, 1.0]

drone 0 → [1.2, -0.75, 1.0]
drone 1 → [1.2,  0.00, 1.0]   (4 drones, 0.5 m spacing along y)
drone 2 → [1.2, +0.75, 1.0]
drone 3 → [1.2, +1.50, 1.0]
```

Slots are matched to drones in current along-axis order so transit paths stay parallel and never cross (`_assign_no_cross`).

**`mode="ring"` — spread onto the loiter orbit before Loiter begins**

Each drone is sent to the ring slot nearest its current bearing (`_assign_ring`). Because paths from a one-sided approach can cross in 2D, each drone flies at a slightly different altitude layer during this leg; Loiter then pulls them back to a common altitude.

### Recon (swarm)

Each drone runs a full-disk search pattern with a rotated starting angle (`phase_offset = 2π·k/N`), so the swarm fans out from different directions simultaneously. Altitude layers keep interlaced paths vertically deconflicted.

**`mode="lawnmower"` (default) — staggered square spirals**

```
drone 0: square spiral, phase_offset = 0
drone 1: square spiral, phase_offset = 2π/N
drone 2: square spiral, phase_offset = 4π/N
...        each drone expands outward from center with right-angle turns
```

**`mode="spiral"` — staggered Archimedean spirals**

```
drone 0: spiral, phase_offset = 0
drone 1: spiral, phase_offset = 2π/N
drone 2: spiral, phase_offset = 4π/N
...        each drone expands outward from center with smooth arcs
```

### Loiter (swarm)

Each drone enters the orbit at its **current bearing** from the target — no explicit phase assignment. Because the preceding `ring` Transit already spread the drones evenly around the orbit, they naturally arrive at equally spaced angles and maintain that spacing while orbiting. All drones share the same target (callable), so the orbit center follows the target if it moves.

### Strike (swarm)

The drone **closest to the target** at the moment the strike phase begins is selected as the sole striker; the rest hold position (IDLE). This avoids the separation filter blocking simultaneous convergences from multiple drones.

```
striker  → the closest drone dashes straight onto the target
others   → hold in place at their loiter positions
```

`relax_safety=True` is set on the striker's setpoints so the separation filter does not brake the terminal dash.

### Swarm phase summary

| Phase | Allocation | Deconfliction |
|---|---|---|
| Transit (formation) | N formation slots from 1 target | slot assigned by along-axis order |
| Recon | staggered full-disk patterns (lawnmower or spiral) | altitude layer per drone |
| Transit (ring) | N ring slots on loiter orbit | slot assigned by bearing + altitude layers |
| Loiter | same params for all (current bearing as entry) | orbit spread from ring transit |
| Strike | closest drone strikes; others hold (IDLE) | single convergence avoids separation filter |

---

## Running the Tactical Examples

### Single drone: `tactical.py`

```sh
cd gym_pybullet_drones/examples/
python tactical.py
```

The full mission runs in sequence: **Transit → Recon → Transit → Loiter → Strike**.
Each phase transition is printed to the terminal with its timestamp. A red sphere marks the (slowly drifting) target; a blue sphere marks the recon search center.

**Options:**

| Flag | Default | Choices |
|---|---|---|
| `--behavior` | `all` | `all` \| `transit` \| `recon` \| `loiter` \| `strike` |
| `--pattern` | `lawnmower` | `lawnmower` \| `spiral` |
| `--gui` | `True` | `True` \| `False` |
| `--plot` | `True` | `True` \| `False` |
| `--duration_sec` | `0` | `0` = auto per behavior |
| `--record_video` | `False` | `True` \| `False` |
| `--output_folder` | `results` | any path |

```sh
python tactical.py --behavior recon --pattern spiral
python tactical.py --behavior all --gui False --plot False   # headless
python tactical.py --behavior strike --duration_sec 15
```

### Multi-drone swarm: `tactical_swarm.py`

```sh
cd gym_pybullet_drones/examples/
python tactical_swarm.py --num_drones 4
```

Each drone is assigned its own sub-task per phase; paths are traced in distinct colors. A red sphere marks the (slowly drifting) target; a blue sphere marks the recon search center.

**`--behavior all` full sequence:**

```
1. Transit  (formation)   line-abreast to the search area          [transit_mode fixed: formation]
2. Recon    (lawnmower/spiral) staggered full-disk search patterns  [recon_mode selectable]
3. Transit  (ring)        spread onto the loiter orbit ring         [always ring, not selectable]
4. Loiter                 orbit the target, evenly spaced
5. Strike                 simultaneous multi-angle dash onto target
```

**Options:**

| Flag | Default | Choices | Applies to |
|---|---|---|---|
| `--num_drones` | `4` | any integer ≥ 2 | all |
| `--behavior` | `all` | `all` \| `transit` \| `recon` \| `loiter` \| `strike` | — |
| `--transit_mode` | `formation` | `formation` \| `ring` | `--behavior transit` only |
| `--recon_mode` | `lawnmower` | `lawnmower` \| `spiral` | `--behavior recon` and `all` |
| `--duration_sec` | `0` | `0` = auto per behavior | all |
| `--gui` | `True` | `True` \| `False` | all |
| `--plot` | `True` | `True` \| `False` | all |

> **Note:** `--transit_mode` only takes effect when `--behavior transit` is used in isolation.
> In `--behavior all`, the two transits are fixed: the first is always `formation`, the second is always `ring`.

```sh
python tactical_swarm.py --num_drones 4
python tactical_swarm.py --num_drones 4 --recon_mode spiral
python tactical_swarm.py --behavior transit --transit_mode ring --num_drones 3
python tactical_swarm.py --behavior recon --recon_mode spiral --num_drones 4
python tactical_swarm.py --num_drones 4 --gui False          # headless
```

---

---

<!-- ================================================================== -->
<!-- Original gym-pybullet-drones README (upstream: learnsyslab/gym-pybullet-drones) -->
<!-- ================================================================== -->

> [!TIP]
> For research work with **symbolic dynamics and constraints**, also try [`safe-control-gym`](https://github.com/learnsyslab/safe-control-gym)
>
> For GPU-accelerated, **differentiable, JAX-based simulation**, also try [`crazyflow`](https://github.com/learnsyslab/crazyflow)
>
> For production-grade deployment of **ROS2 + PX4/ArduPilot + YOLO/LiDAR**, use [`aerial-autonomy-stack`](https://github.com/JacopoPan/aerial-autonomy-stack)

# gym-pybullet-drones

This is a minimalist refactoring of the original `gym-pybullet-drones` repository, designed for compatibility with [`gymnasium`](https://github.com/Farama-Foundation/Gymnasium), [`stable-baselines3` 2.0](https://github.com/DLR-RM/stable-baselines3/pull/1327), and [`betaflight`](https://github.com/betaflight/betaflight)/[`crazyflie-firmware`](https://github.com/bitcraze/crazyflie-firmware/) SITL.

> **NEWS**: `gym-pybullet-drones` was featured in [GitHub's Maintainer Spotlight 2026](https://maintainermonth.github.com/academia/gym-pybullet-drones-maintainer-spotlight)

> **NOTE**: if you want to access the original codebase, presented at IROS in 2021, please `git checkout [paper|master]`

<img src="gym_pybullet_drones/assets/helix.gif" alt="formation flight" width="325"> <img src="gym_pybullet_drones/assets/helix.png" alt="control info" width="425">

## Installation

Tested on Intel x64/Ubuntu 22.04 and Apple Silicon/macOS 26.2.

```sh
git clone https://github.com/Giho-Kim/minimax_drones.git
cd minimax_drones/

conda create -n drones python=3.10
conda activate drones

pip3 install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`

# check installed packages with `conda list`, deactivate with `conda deactivate`, remove with `conda remove -n drones --all`
```

## Use

### PID control examples

```sh
cd gym_pybullet_drones/examples/
python3 pid.py # position and velocity reference
python3 pid_velocity.py # desired velocity reference
```

### Downwash effect example

```sh
cd gym_pybullet_drones/examples/
python3 downwash.py
```

### Reinforcement learning examples (SB3's PPO)

```sh
cd gym_pybullet_drones/examples/
python learn.py # task: single drone hover at z == 1.0
python learn.py --multiagent true # task: 2-drone hover at z == 1.2 and 0.7

LATEST_MODEL=$(ls -t results | head -n 1) && python play.py --model_path "results/${LATEST_MODEL}/best_model.zip" # play and visualize the most recent learned policy after training
```

<img src="gym_pybullet_drones/assets/rl.gif" alt="rl example" width="375"> <img src="gym_pybullet_drones/assets/marl.gif" alt="marl example" width="375">

### Run all tests

```sh
# from the repo's top folder
cd gym-pybullet-drones/
pytest tests/
```

### Betaflight SITL example (Ubuntu only)

```sh
git clone https://github.com/betaflight/betaflight 
cd betaflight/
git checkout cafe727 # `master` branch head at the time of writing (future release 4.5)
make arm_sdk_install # if needed, `apt install curl``
make TARGET=SITL # comment out line: https://github.com/betaflight/betaflight/blob/master/src/main/main.c#L52
cp ~/gym-pybullet-drones/gym_pybullet_drones/assets/eeprom.bin ~/betaflight/ # assuming both gym-pybullet-drones/ and betaflight/ were cloned in ~/
betaflight/obj/main/betaflight_SITL.elf
```

In another terminal, run the example

```sh
conda activate drones
cd gym_pybullet_drones/examples/
python3 beta.py --num_drones 1 # check the steps in the file's docstrings to use multiple drones
```

### `pycffirmware` Python Bindings example (multiplatform, single-drone)

First, install [`pycffirmware`](https://github.com/learnsyslab/pycffirmware?tab=readme-ov-file#installation) for Ubuntu, macOS, or Windows, then

```sh
cd gym_pybullet_drones/examples/
python3 cf.py
```

## Citation

If you wish, please cite our [IROS 2021 paper](https://arxiv.org/abs/2103.02142) ([and original codebase](https://github.com/learnsyslab/gym-pybullet-drones/tree/paper)) as

```bibtex
@INPROCEEDINGS{panerati2021learning,
      title={Learning to Fly---a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control}, 
      author={Jacopo Panerati and Hehui Zheng and SiQi Zhou and James Xu and Amanda Prorok and Angela P. Schoellig},
      booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      year={2021},
      volume={},
      number={},
      pages={7512-7519},
      doi={10.1109/IROS51168.2021.9635857}
}
```

## References

- Erwin Coumans and Yunfei Bai (2023) [*PyBullet Quickstart Guide*](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit?tab=t.0#heading=h.2ye70wns7io3)
- Carlos Luis and Jeroome Le Ny (2016) [*Design of a Trajectory Tracking Controller for a Nanoquadcopter*](https://arxiv.org/pdf/1608.05786.pdf)
- Nathan Michael, Daniel Mellinger, Quentin Lindsey, Vijay Kumar (2010) [*The GRASP Multiple Micro-UAV Testbed*](https://ieeexplore.ieee.org/document/5569026)
- Benoit Landry (2014) [*Planning and Control for Quadrotor Flight through Cluttered Environments*](http://groups.csail.mit.edu/robotics-center/public_papers/Landry15)
- Julian Forster (2015) [*System Identification of the Crazyflie 2.0 Nano Quadrocopter*](https://www.research-collection.ethz.ch/handle/20.500.11850/214143)
- Antonin Raffin, Ashley Hill, Maximilian Ernestus, Adam Gleave, Anssi Kanervisto, and Noah Dormann (2019) [*Stable Baselines3*](https://github.com/DLR-RM/stable-baselines3)
- Guanya Shi, Xichen Shi, Michael O'Connell, Rose Yu, Kamyar Azizzadenesheli, Animashree Anandkumar, Yisong Yue, and Soon-Jo Chung (2019)
[*Neural Lander: Stable Drone Landing Control Using Learned Dynamics*](https://arxiv.org/pdf/1811.08027.pdf)
- C. Karen Liu and Dan Negrut (2020) [*The Role of Physics-Based Simulators in Robotics*](https://www.annualreviews.org/doi/pdf/10.1146/annurev-control-072220-093055)
- Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio, and Davide Scaramuzza (2020) [*Flightmare: A Flexible Quadrotor Simulator*](https://arxiv.org/pdf/2009.00563.pdf)

-----
> UTIAS / [Learning Systems and Robotics Lab](https://github.com/learnsyslab) / [Vector Institute](https://github.com/VectorInstitute) / University of Cambridge's [Prorok Lab](https://github.com/proroklab)
