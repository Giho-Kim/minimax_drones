"""Render N drones cooperating on one tactical mission.

The multi-drone counterpart of `tactical.py`. A single `SwarmCoordinator`
sequences one shared mission and splits each macro across the swarm:

    (1) Transit  fly out in a line-abreast formation
    (2) Recon    split the search disk into parallel bands, one per drone
    (3) Transit  reposition (formation) to a standoff near the target
    (4) Loiter   orbit the target, evenly spread in phase (multi-angle watch)
    (5) Strike   all dash onto the target from their distinct loiter bearings

A reactive separation filter keeps the drones apart (relaxed for strikers). The
flown paths are traced, one color per drone, and a red marker shows the moving
target. There is no reward -- the goal is to verify, by rendering, that the
swarm cooperates as intended.

Example
-------
In a terminal, run as:

    $ python tactical_swarm.py --num_drones 4

"""
import time
import argparse

import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics, BehaviorType
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.swarm import SwarmCoordinator
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 4
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = True
DEFAULT_PARTITION = 'band'          # recon split scheme: band | sector
DEFAULT_BEHAVIOR = 'all'
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 0            # 0 = auto per behavior
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
PARTITION_CHOICES = ['band', 'sector']
BEHAVIOR_CHOICES = ['all', 'transit', 'recon', 'loiter', 'strike']
AUTO_DURATION = {'all': 50, 'transit': 12, 'recon': 25, 'loiter': 15, 'strike': 15}

# Fraction of each behavior's velocity feed-forward passed to the PID. The tiny
# CF2X tends to tip and lose altitude when the feed-forward flips sharply (e.g.
# lawnmower turns) under the extra perturbation of the swarm separation filter,
# so we damp it. Position control still tracks the setpoint; this only softens
# the velocity term.
FEEDFORWARD = 0.5

# Distinct trace color per drone (cycled if there are more drones).
DRONE_COLORS = [
    [1.0, 0.2, 0.2], [0.2, 0.5, 1.0], [0.2, 0.8, 0.2], [1.0, 0.7, 0.0],
    [0.7, 0.2, 1.0], [0.0, 0.8, 0.8], [1.0, 0.4, 0.7], [0.6, 0.6, 0.6],
]

# Moving target: an elevated target that drifts slowly in +x. Keeping it off
# the ground means the terminal strike does not overshoot into the floor.
TARGET_START = np.array([2.2, 1.0, 0.5])
TARGET_VEL = np.array([0.02, 0.0, 0.0])


def target_at(t_sim):
    """World position of the surveilled/struck target at simulation time t."""
    return TARGET_START + TARGET_VEL * t_sim


class _Clock:
    """Holder so the moving-target callables can read the global sim time."""
    shared = 0.0


def build_mission(clock, partition, behavior='all'):
    """Cooperative mission: one macro per phase, split across the swarm.

    Parameters
    ----------
    behavior : str
        ``"all"`` runs the full sequence; otherwise one of
        ``transit``/``recon``/``loiter``/``strike`` runs that single phase.
    """
    def tgt(_unused, _clock=clock):
        return target_at(_clock.shared)

    recon_pattern = "spiral" if partition == "sector" else "lawnmower"
    orbit_radius = 0.6
    orbit_alt = float(TARGET_START[2] + orbit_radius)

    transit1 = {"type": BehaviorType.TRANSIT, "mode": "formation",
                "params": {"target": [1.2, 0.0, 1.0], "spacing": 0.5,
                           "v_max": 0.5, "a_max": 0.6}}
    recon    = {"type": BehaviorType.RECON, "mode": partition,
                "params": {"center": [1.2, 1.0, 1.0], "radius": 0.7,
                           "pattern": recon_pattern, "swath": 0.4, "speed": 0.4}}
    transit2 = {"type": BehaviorType.TRANSIT, "mode": "ring",
                "params": {"center": [TARGET_START[0], TARGET_START[1], orbit_alt],
                           "radius": orbit_radius, "v_max": 0.5, "a_max": 0.6}}
    loiter   = {"type": BehaviorType.LOITER, "mode": None,
                "params": {"target": tgt, "standoff_alt": orbit_radius,
                           "radius": orbit_radius, "orbit_speed": 0.4, "duration": 8.0}}
    strike   = {"type": BehaviorType.STRIKE, "mode": None,
                "params": {"target": tgt, "dash_speed": 1.0, "hit_radius": 0.12,
                           "ring": 0.4, "decel_dist": 0.5}}

    full = [transit1, recon, transit2, loiter, strike]
    single = {'transit': [transit1], 'recon': [recon],
              'loiter': [loiter], 'strike': [strike]}
    return full if behavior == 'all' else single[behavior]


def _init_xyzs(behavior, num_drones):
    """Return (N, 3) spawn positions appropriate for the chosen behavior."""
    if behavior in ('all', 'transit'):
        return np.array([[0.6 * (i - (num_drones - 1) / 2.0), 0.0, 0.1]
                         for i in range(num_drones)])
    if behavior == 'recon':
        return np.array([[0.5 * (i - (num_drones - 1) / 2.0) + 1.2, 0.0, 1.0]
                         for i in range(num_drones)])
    # loiter / strike: spread evenly on the orbit ring above the target
    orbit_radius = 0.6
    orbit_alt = float(TARGET_START[2] + orbit_radius)
    angles = [2.0 * np.pi * k / num_drones for k in range(num_drones)]
    return np.array([[TARGET_START[0] + orbit_radius * np.cos(a),
                      TARGET_START[1] + orbit_radius * np.sin(a),
                      orbit_alt]
                     for a in angles])


def run(drone=DEFAULT_DRONE, num_drones=DEFAULT_NUM_DRONES, physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI, record_video=DEFAULT_RECORD_VIDEO, plot=DEFAULT_PLOT,
        partition=DEFAULT_PARTITION, behavior=DEFAULT_BEHAVIOR,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC, output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB):

    if duration_sec in (0, None):
        duration_sec = AUTO_DURATION[behavior]

    INIT_XYZS = _init_xyzs(behavior, num_drones)
    INIT_RPYS = np.zeros((num_drones, 3))

    env = CtrlAviary(drone_model=drone,
                     num_drones=num_drones,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=physics,
                     neighbourhood_radius=10,
                     pyb_freq=simulation_freq_hz,
                     ctrl_freq=control_freq_hz,
                     gui=gui,
                     record=record_video,
                     obstacles=False,
                     user_debug_gui=False,
                     output_folder=output_folder)
    PYB_CLIENT = env.getPyBulletClient()

    #### Swarm coordinator + per-drone PID #####################
    clock = _Clock()
    coordinator = SwarmCoordinator(num_drones=num_drones,
                                   ctrl_freq=control_freq_hz,
                                   mission=build_mission(clock, partition, behavior))
    ctrl = [DSLPIDControl(drone_model=drone) for _ in range(num_drones)]

    #### Visual-only target marker #############################
    target_marker = None
    if gui:
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.08,
                                  rgbaColor=[1, 0, 0, 1], physicsClientId=PYB_CLIENT)
        target_marker = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                          baseVisualShapeIndex=vis,
                                          basePosition=TARGET_START.tolist(),
                                          physicsClientId=PYB_CLIENT)

    #### Logger (preallocated so the full run logs cleanly) ####
    logger = Logger(logging_freq_hz=control_freq_hz, num_drones=num_drones,
                    duration_sec=duration_sec, output_folder=output_folder,
                    colab=colab)

    #### Run the simulation ####################################
    action = np.zeros((num_drones, 4))
    prev_pos = INIT_XYZS.copy()
    prev_phase = None
    START = time.time()
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        clock.shared = i / env.CTRL_FREQ

        #### Step the simulation ###############################
        obs, _, _, _, _ = env.step(action)

        #### Swarm coordinator -> per-drone setpoints ##########
        setpoints = coordinator.step(obs)

        #### Announce phase transitions ########################
        macro = coordinator.current_macro
        phase = macro["type"].value if macro else None
        if phase != prev_phase:
            print(f"[{clock.shared:5.1f}s] phase -> {str(phase).upper()}")
            prev_phase = phase

        #### Low-level PID tracking, per drone #################
        for k in range(num_drones):
            action[k, :], _, _ = ctrl[k].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[k],
                target_pos=setpoints[k].pos,
                target_rpy=setpoints[k].rpy,
                target_vel=FEEDFORWARD * setpoints[k].vel,
            )

        #### Visualization #####################################
        if gui:
            tgt = target_at(clock.shared)
            p.resetBasePositionAndOrientation(target_marker, tgt.tolist(),
                                              [0, 0, 0, 1], physicsClientId=PYB_CLIENT)
            for k in range(num_drones):
                cur = obs[k][0:3]
                p.addUserDebugLine(prev_pos[k].tolist(), cur.tolist(),
                                   lineColorRGB=DRONE_COLORS[k % len(DRONE_COLORS)],
                                   lineWidth=2, lifeTime=0, physicsClientId=PYB_CLIENT)
                prev_pos[k] = cur.copy()

        #### Log ###############################################
        for k in range(num_drones):
            logger.log(drone=k, timestamp=clock.shared, state=obs[k],
                       control=np.hstack([setpoints[k].pos, setpoints[k].rpy, np.zeros(6)]))

        env.render()
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    env.close()
    logger.save_as_csv("tactical_swarm")
    if plot:
        logger.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Render a cooperative N-drone tactical mission.')
    parser.add_argument('--drone', default=DEFAULT_DRONE, type=DroneModel,
                        help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones', default=DEFAULT_NUM_DRONES, type=int,
                        help='Number of drones (default: 4)', metavar='')
    parser.add_argument('--physics', default=DEFAULT_PHYSICS, type=Physics,
                        help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool,
                        help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool,
                        help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool,
                        help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--behavior', default=DEFAULT_BEHAVIOR, type=str, choices=BEHAVIOR_CHOICES,
                        help='Which behavior to run: all | transit | recon | loiter | strike (default: all)', metavar='')
    parser.add_argument('--partition', default=DEFAULT_PARTITION, type=str, choices=PARTITION_CHOICES,
                        help='Recon split scheme: band | sector (default: band)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,
                        help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int,
                        help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int,
                        help='Duration in seconds (default: 0 = auto per behavior)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool,
                        help='Whether example is being run by a notebook (default: False)', metavar='')
    ARGS = parser.parse_args()
    run(**vars(ARGS))
