"""Hand-coded macro behaviors and the FSM that sequences them.

These implement the four core tactical behaviors as point-to-point trajectory
generators on top of the existing `DSLPIDControl` low-level controller:

    (1) Transit  -- straight line-of-sight move with a trapezoidal speed profile
    (2) Recon    -- lawnmower / spiral coverage of a disk
    (3) Loiter   -- circular orbit that tracks a (moving) target
    (4) Strike   -- minimum-time terminal dash onto a target

The :class:`BehaviorManager` exposes the RL-facing command interface and emits
one :class:`Setpoint` per control step.
"""
from gym_pybullet_drones.behaviors.base_behavior import BaseBehavior, Setpoint
from gym_pybullet_drones.behaviors.transit import Transit
from gym_pybullet_drones.behaviors.recon import Recon
from gym_pybullet_drones.behaviors.loiter import Loiter
from gym_pybullet_drones.behaviors.strike import Strike
from gym_pybullet_drones.behaviors.fsm import BehaviorManager

__all__ = [
    "BaseBehavior",
    "Setpoint",
    "Transit",
    "Recon",
    "Loiter",
    "Strike",
    "BehaviorManager",
]
