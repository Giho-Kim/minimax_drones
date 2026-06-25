"""Render the four hand-coded macro behaviors with an FSM mission.

This mirrors `pid.py` -- a `CtrlAviary` simulation tracked by `DSLPIDControl` --
but the per-step position/velocity setpoints come from the macro-behavior FSM
in `gym_pybullet_drones.behaviors` instead of a pre-baked waypoint array.

The scripted mission exercises every behavior in sequence:

    (1) Transit  fly out to a start point
    (2) Recon    lawnmower-search a disk
    (3) Transit  reposition to a standoff
    (4) Loiter   orbit and track a (slowly moving) target
    (5) Strike   terminal dash onto that target

A red marker shows the (moving) target; the flown path is traced and colored by
the active behavior. There is no reward here -- the point is to verify, by
rendering, that the four abstractions behave as intended.

Example
-------
In a terminal, run as:

    $ python tactical.py

"""
import time
import argparse

import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics, BehaviorType
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.behaviors import BehaviorManager
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 0   # 0 = auto-pick a duration that fits the behavior
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_BEHAVIOR = 'all'
BEHAVIOR_CHOICES = ['all', 'transit', 'recon', 'loiter', 'strike']
DEFAULT_PATTERN = 'lawnmower'
PATTERN_CHOICES = ['lawnmower', 'spiral']
DEFAULT_STRIKE_MODE = 'guided'
STRIKE_MODE_CHOICES = ['guided', 'terminal']
DEFAULT_VIEW = 'top'
VIEW_CHOICES = ['top', 'iso', 'side', 'follow']
VIEW_PRESETS = {
    'top':  (4.0,   0, -89.9),
    'iso':  (4.0,  45,   -35),
    'side': (4.0,   0,   -15),
}

# Auto durations (s) used when --duration_sec is 0, sized to each mission so
# single-behavior demos stay snappy while the full sequence has room to finish.
AUTO_DURATION = {'all': 45, 'transit': 8, 'recon': 18, 'loiter': 12, 'strike': 12}

# Path-trace color per behavior (RGB), purely for visualization.
TRACE_COLOR = {
    BehaviorType.IDLE:    [0.5, 0.5, 0.5],
    BehaviorType.TRANSIT: [0.0, 0.4, 1.0],
    BehaviorType.RECON:   [0.0, 0.8, 0.2],
    BehaviorType.LOITER:  [1.0, 0.6, 0.0],
    BehaviorType.STRIKE:  [1.0, 0.0, 0.0],
}

# Moving target: starts here and drifts slowly in +x during loiter/strike.
TARGET_START = np.array([0.0, 0.0, 0.5])
TARGET_VEL = np.array([0.06, 0.0, 0.0])
RECON_CENTER = np.array([1.5, 1.5, 1.0])


def target_at(t_sim):
    """World position of the surveilled/struck target at simulation time t."""
    return TARGET_START + TARGET_VEL * t_sim


class _Clock:
    """Tiny holder so behavior target callables can read the global sim time."""
    shared = 0.0


def run(drone=DEFAULT_DRONE, physics=DEFAULT_PHYSICS, gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VIDEO, plot=DEFAULT_PLOT,
        behavior=DEFAULT_BEHAVIOR, pattern=DEFAULT_PATTERN, view=DEFAULT_VIEW,
        strike_mode=DEFAULT_STRIKE_MODE,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC, output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB):

    if duration_sec in (0, None):
        duration_sec = AUTO_DURATION[behavior]

    #### Behavior FSM (also fixes the spawn pose for the chosen behavior) ####
    clock = _Clock()
    manager = BehaviorManager(ctrl_freq=control_freq_hz)
    INIT_XYZS = build_mission(manager, clock, behavior, pattern, strike_mode)

    #### Initialize the simulation #############################
    INIT_RPYS = np.array([[0.0, 0.0, 0.0]])

    env = CtrlAviary(drone_model=drone,
                     num_drones=1,
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

    if gui and view in VIEW_PRESETS:
        dist, yaw, pitch = VIEW_PRESETS[view]
        p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=yaw,
                                     cameraPitch=pitch,
                                     cameraTargetPosition=TARGET_START.tolist(),
                                     physicsClientId=PYB_CLIENT)

    #### Low-level PID #########################################
    ctrl = DSLPIDControl(drone_model=drone)

    #### Visual-only markers ###################################
    target_marker = None
    recon_marker = None
    if gui:
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.08,
                                  rgbaColor=[1, 0, 0, 1], physicsClientId=PYB_CLIENT)
        target_marker = p.createMultiBody(baseMass=0,
                                          baseCollisionShapeIndex=-1,
                                          baseVisualShapeIndex=vis,
                                          basePosition=TARGET_START.tolist(),
                                          physicsClientId=PYB_CLIENT)
    if gui:
        recon_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.08,
                                        rgbaColor=[0, 0.5, 1, 1], physicsClientId=PYB_CLIENT)
        recon_marker = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                         baseVisualShapeIndex=recon_vis,
                                         basePosition=RECON_CENTER.tolist(),
                                         physicsClientId=PYB_CLIENT)

    #### Logger (preallocated so the full run logs cleanly) ####
    logger = Logger(logging_freq_hz=control_freq_hz, num_drones=1,
                    duration_sec=duration_sec, output_folder=output_folder,
                    colab=colab)

    #### Run the simulation ####################################
    # The loop runs the full duration; once the mission finishes the FSM holds
    # position in IDLE, which keeps the logs/plots well-formed.
    action = np.zeros((1, 4))
    prev_pos = INIT_XYZS[0].copy()
    prev_type = None
    strike_body = None  # rigid body spawned only when strike begins
    strike_hit = False  # becomes True the moment drone contacts the strike body
    START = time.time()
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        clock.shared = i / env.CTRL_FREQ

        #### Step the simulation ###############################
        obs, _, _, _, _ = env.step(action)
        state = obs[0]

        #### Macro behavior -> setpoint ########################
        sp = manager.step(state)

        #### Announce behavior transitions #####################
        if manager.current_type != prev_type:
            print(f"[{clock.shared:5.1f}s] -> {manager.current_type.value.upper()}")
            prev_type = manager.current_type

        #### Low-level PID tracking ############################
        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=state,
            target_pos=sp.pos,
            target_rpy=sp.rpy,
            target_vel=sp.vel,
        )

        #### Strike: rigid target tracks path until contact #####
        if manager.current_type == BehaviorType.STRIKE:
            if strike_body is None:
                s_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.08,
                                               physicsClientId=PYB_CLIENT)
                s_vis = (p.createVisualShape(p.GEOM_SPHERE, radius=0.08,
                                             rgbaColor=[1, 0, 0, 1],
                                             physicsClientId=PYB_CLIENT) if gui else -1)
                strike_body = p.createMultiBody(baseMass=0.1,
                                                baseCollisionShapeIndex=s_col,
                                                baseVisualShapeIndex=s_vis,
                                                basePosition=target_at(clock.shared).tolist(),
                                                physicsClientId=PYB_CLIENT)
                p.changeDynamics(strike_body, -1, linearDamping=0.9, angularDamping=0.9,
                                 physicsClientId=PYB_CLIENT)
                if gui:
                    p.resetBasePositionAndOrientation(target_marker, [1000, 0, 0],
                                                      [0, 0, 0, 1], physicsClientId=PYB_CLIENT)
            if not strike_hit:
                if getattr(manager.current, 'impact', False):
                    strike_hit = True
                    drone_vel = np.array(state[10:13])
                    # Knock target in the direction the drone was flying
                    p.resetBaseVelocity(strike_body, drone_vel.tolist(), [0, 0, 0],
                                        physicsClientId=PYB_CLIENT)
                    # Knock drone back (opposite direction + downward to overcome PID)
                    knock = (-drone_vel * 1.5 - np.array([0, 0, 2.0])).tolist()
                    p.resetBaseVelocity(env.getDroneIds()[0], knock, [0, 0, 0],
                                        physicsClientId=PYB_CLIENT)
                else:
                    tgt_pos = target_at(clock.shared)
                    p.resetBasePositionAndOrientation(strike_body, tgt_pos.tolist(),
                                                      [0, 0, 0, 1], physicsClientId=PYB_CLIENT)
                    p.resetBaseVelocity(strike_body, [0, 0, 0], [0, 0, 0],
                                        physicsClientId=PYB_CLIENT)

        #### Visualization #####################################
        if gui:
            if manager.current_type != BehaviorType.STRIKE:
                tgt = target_at(clock.shared)
                p.resetBasePositionAndOrientation(target_marker, tgt.tolist(),
                                                  [0, 0, 0, 1], physicsClientId=PYB_CLIENT)
            cur_pos = state[0:3]
            p.addUserDebugLine(prev_pos.tolist(), cur_pos.tolist(),
                               lineColorRGB=TRACE_COLOR[manager.current_type],
                               lineWidth=2, lifeTime=0, physicsClientId=PYB_CLIENT)
            prev_pos = cur_pos.copy()
            if view == 'follow':
                p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=0,
                                             cameraPitch=-30,
                                             cameraTargetPosition=cur_pos.tolist(),
                                             physicsClientId=PYB_CLIENT)

        #### Log ###############################################
        logger.log(drone=0, timestamp=clock.shared, state=state,
                   control=np.hstack([sp.pos, sp.rpy, np.zeros(6)]))

        env.render()
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    env.close()
    logger.save_as_csv("tactical")
    if plot:
        logger.plot()


def build_mission(manager, clock, behavior="all", pattern="lawnmower",
                  strike_mode="guided"):
    """Queue the mission and return the matching initial position.

    Parameters
    ----------
    manager : BehaviorManager
        The FSM to program.
    clock : _Clock
        Shared simulation clock, bound into the moving-target callables.
    behavior : str
        ``"all"`` runs the full Transit->Recon->Transit->Loiter->Strike
        sequence; otherwise one of ``transit``/``recon``/``loiter``/``strike``
        runs that single behavior in isolation, with the drone spawned at a
        sensible starting pose so it is clearly visible.
    pattern : str
        Recon search pattern, ``"lawnmower"`` or ``"spiral"``.
    strike_mode : str
        Strike dash style: ``"guided"`` (precise) or ``"terminal"`` (committed
        high-speed ballistic dive).

    Returns
    -------
    np.ndarray
        (1, 3) initial XYZ to spawn the drone at for the chosen behavior.
    """
    def tgt(_unused, _clock=clock):
        return target_at(_clock.shared)

    # (init position, [(BehaviorType, params), ...]) per single-behavior mode.
    transit = (BehaviorType.TRANSIT, dict(target=[1.5, 0.0, 1.0], v_max=0.7, a_max=1.0))
    recon = (BehaviorType.RECON, dict(center=RECON_CENTER.tolist(), radius=1.0,
                                      pattern=pattern, swath=0.5, speed=0.6))
    loiter = (BehaviorType.LOITER, dict(target=tgt, standoff_alt=0.8,
                                        depression_deg=45.0, orbit_speed=0.7,
                                        duration=8.0))
    strike = (BehaviorType.STRIKE, dict(target=tgt, mode=strike_mode,
                                        dash_speed=3.0, hit_radius=0.15))

    single = {
        "transit": (np.array([[0.0, 0.0, 0.1]]), [transit]),
        "recon":   (np.array([[1.5, 0.0, 1.0]]), [recon]),
        "loiter":  (np.array([[3.0, 1.4, 1.0]]), [loiter]),
        "strike":  (np.array([[3.0, 1.4, 1.0]]), [strike]),
    }

    if behavior == "all":
        init = np.array([[0.0, 0.0, 0.1]])
        queue = [transit, recon,
                 (BehaviorType.TRANSIT, dict(target=[3.0, 1.4, 1.0], v_max=0.7, a_max=1.0)),
                 loiter, strike]
    else:
        init, queue = single[behavior]

    manager.queue(queue)
    return init


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Render the four macro behaviors (Transit/Recon/Loiter/Strike).')
    parser.add_argument('--drone', default=DEFAULT_DRONE, type=DroneModel,
                        help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
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
    parser.add_argument('--pattern', default=DEFAULT_PATTERN, type=str, choices=PATTERN_CHOICES,
                        help='Recon search pattern: lawnmower | spiral (default: lawnmower)', metavar='')
    parser.add_argument('--view', default=DEFAULT_VIEW, type=str, choices=VIEW_CHOICES,
                        help='Camera view: top | iso | side | follow (default: top)', metavar='')
    parser.add_argument('--strike_mode', default=DEFAULT_STRIKE_MODE, type=str, choices=STRIKE_MODE_CHOICES,
                        help='Strike dash: guided | terminal (default: guided)', metavar='')
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
