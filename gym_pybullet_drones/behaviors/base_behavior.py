"""Abstract base class for the hand-coded macro behaviors.

A behavior is a small, self-contained trajectory generator. It is reset with
the drone's current state (the macro's starting condition) plus a few
parameters, and then queried once per control step. Each query returns a
:class:`Setpoint` that the low-level PID controller (`DSLPIDControl`) can track.

The behaviors are deliberately free of any PyBullet / gym dependency: they only
consume the 20-dimensional state vector produced by `_getDroneStateVector()`
and emit numpy setpoints. This keeps them unit-testable and reusable both from
the rendering demo and, later, from an RL environment.

State vector layout (see `BaseAviary._getDroneStateVector`)::

    [0:3]   position      (x, y, z)
    [3:7]   quaternion    (qx, qy, qz, qw)
    [7:10]  roll/pitch/yaw
    [10:13] linear velocity
    [13:16] angular velocity
    [16:20] last motor RPMs
"""
from dataclasses import dataclass, field

import numpy as np


# Convenient state-vector slices (shared by every behavior).
POS = slice(0, 3)
QUAT = slice(3, 7)
RPY = slice(7, 10)
VEL = slice(10, 13)
YAW = 9


@dataclass
class Setpoint:
    """Reference handed to the low-level PID controller for one control step.

    Attributes
    ----------
    pos : np.ndarray
        (3,) desired position in the world frame.
    rpy : np.ndarray
        (3,) desired roll/pitch/yaw. Only yaw is meaningful for the DSL PID
        position controller; roll/pitch are derived from the thrust vector.
    vel : np.ndarray
        (3,) desired velocity, used as a feed-forward term for smooth tracking.
    relax_safety : bool
        Hint that the safety filter may be (partially) relaxed for this
        setpoint. Only the Strike behavior sets this to ``True``.
    """
    pos: np.ndarray
    rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    relax_safety: bool = False


class BaseBehavior:
    """Common interface and helpers for every macro behavior."""

    #: Human-readable name, overridden by subclasses.
    name = "base"

    def __init__(self, ctrl_freq: int):
        """Parameters
        ----------
        ctrl_freq : int
            Control frequency in Hz. The behavior integrates its internal
            clock with ``dt = 1 / ctrl_freq``.
        """
        self.ctrl_freq = ctrl_freq
        self.dt = 1.0 / ctrl_freq
        self.t = 0.0
        self._done = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def reset(self, state, **params):
        """Initialize the behavior from the current state and parameters.

        Subclasses should call ``super().reset(state, **params)`` first to
        reset the internal clock and the ``done`` flag, then plan their
        trajectory from ``state[POS]``.
        """
        self.t = 0.0
        self._done = False

    def step(self, state) -> Setpoint:
        """Return the :class:`Setpoint` for the current control step.

        Subclasses must override this. Implementations are expected to advance
        ``self.t`` by ``self.dt`` and to set ``self._done`` when finished.
        """
        raise NotImplementedError

    def is_done(self) -> bool:
        """Whether the behavior has completed (triggers the next FSM state)."""
        return self._done

    # ------------------------------------------------------------------ #
    # Shared geometry helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _yaw_towards(frm, to) -> float:
        """Yaw (rad) that points the drone's x-axis from ``frm`` toward ``to``."""
        d = np.asarray(to)[:2] - np.asarray(frm)[:2]
        if np.linalg.norm(d) < 1e-6:
            return 0.0
        return float(np.arctan2(d[1], d[0]))
