"""Finite-state machine that sequences the four macro behaviors.

The :class:`BehaviorManager` is the layer an RL policy (or, here, a scripted
mission) talks to. It receives high-level commands -- ``(BehaviorType, params)``
-- and, once per control step, produces a low-level :class:`Setpoint` by
delegating to the currently active behavior. When a behavior reports ``done``
the FSM pops the next command off the mission queue; with nothing queued it
falls back to an ``IDLE`` hold at the last position.

    RL / mission  --(BehaviorType, params)-->  BehaviorManager (FSM)
                                                     |
                                          Transit / Recon / Loiter / Strike
                                                     |
                                                  Setpoint
                                                     |
                                              DSLPIDControl --> RPMs
"""
from collections import deque

import numpy as np

from gym_pybullet_drones.utils.enums import BehaviorType
from gym_pybullet_drones.behaviors.base_behavior import BaseBehavior, Setpoint, POS
from gym_pybullet_drones.behaviors.transit import Transit
from gym_pybullet_drones.behaviors.recon import Recon
from gym_pybullet_drones.behaviors.loiter import Loiter
from gym_pybullet_drones.behaviors.strike import Strike


class _Hold(BaseBehavior):
    """IDLE behavior: hover in place at the position captured on reset."""

    name = "idle"

    def reset(self, state, **_):
        super().reset(state)
        self.pos = np.array(state[POS], dtype=float)
        self.yaw = float(state[9])

    def step(self, state) -> Setpoint:
        # Never reports done; the FSM stays here until a new command arrives.
        return Setpoint(pos=self.pos, rpy=np.array([0.0, 0.0, self.yaw]),
                        vel=np.zeros(3))


class BehaviorManager:
    """Sequences macro behaviors and emits per-step setpoints."""

    def __init__(self, ctrl_freq: int):
        self.ctrl_freq = ctrl_freq
        # One reusable instance per behavior type.
        self._registry = {
            BehaviorType.IDLE: _Hold(ctrl_freq),
            BehaviorType.TRANSIT: Transit(ctrl_freq),
            BehaviorType.RECON: Recon(ctrl_freq),
            BehaviorType.LOITER: Loiter(ctrl_freq),
            BehaviorType.STRIKE: Strike(ctrl_freq),
        }
        self.mission = deque()
        self.current_type = BehaviorType.IDLE
        self.current = self._registry[BehaviorType.IDLE]
        self._started = False

    # ------------------------------------------------------------------ #
    # Mission programming
    # ------------------------------------------------------------------ #
    def command(self, behavior_type: BehaviorType, **params):
        """Enqueue a single macro command. Returns ``self`` for chaining."""
        self.mission.append((behavior_type, params))
        return self

    def queue(self, items):
        """Enqueue a list of ``(BehaviorType, params)`` commands."""
        for behavior_type, params in items:
            self.command(behavior_type, **params)
        return self

    # ------------------------------------------------------------------ #
    # Per-step update
    # ------------------------------------------------------------------ #
    def _activate_next(self, state):
        """Pop and reset the next queued behavior, or fall back to IDLE."""
        if self.mission:
            behavior_type, params = self.mission.popleft()
            self.current_type = behavior_type
            self.current = self._registry[behavior_type]
            self.current.reset(state, **params)
        else:
            self.current_type = BehaviorType.IDLE
            self.current = self._registry[BehaviorType.IDLE]
            self.current.reset(state)

    def step(self, state) -> Setpoint:
        """Advance the FSM by one control step and return a setpoint."""
        # Lazily start the first queued command (reset from the real state).
        if not self._started:
            self._activate_next(state)
            self._started = True

        # Advance the FSM. We move on when (a) the active behavior finished and
        # more work is queued, (b) the active behavior finished with nothing
        # queued (settle into IDLE), or (c) we are holding in IDLE and a new
        # command has been queued (re-tasking, e.g. by the swarm coordinator).
        # Loop so already-satisfied behaviors are skipped within a single step.
        guard = 0
        while guard <= len(self._registry) + 2:
            done = self.current.is_done()
            idle = self.current_type == BehaviorType.IDLE
            if (done and self.mission) or (idle and self.mission) \
                    or (done and not idle):
                self._activate_next(state)
                guard += 1
            else:
                break

        return self.current.step(state)

    @property
    def finished(self) -> bool:
        """True once the queue is empty and the FSM has settled into IDLE."""
        return self._started and not self.mission \
            and self.current_type == BehaviorType.IDLE
