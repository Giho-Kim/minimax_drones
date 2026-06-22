"""(4) Strike -- terminal dash onto a target.

RL command
    "Strike detected target T."

Abstraction
    Switch to a maximum-acceleration terminal dash that exceeds the cruise
    speed. The safety filter constraints are (partially) relaxed -- signaled by
    ``Setpoint.relax_safety`` -- and the trajectory drives the position error
    between the drone and the target to zero in minimum time. The dash speed is
    capped by ``dash_speed`` and commanded as a strong velocity feed-forward
    along the line of sight, while the position setpoint is pinned to the
    target so the controller converges precisely at impact.
"""
import numpy as np

from gym_pybullet_drones.behaviors.base_behavior import BaseBehavior, Setpoint, POS


class Strike(BaseBehavior):
    """Minimum-time terminal guidance onto a static or moving target."""

    name = "strike"

    def reset(self, state, target, dash_speed=1.6, hit_radius=0.12,
              timeout=10.0, decel_dist=0.4, **_):
        """Arm the dash toward ``target``.

        Parameters
        ----------
        state : np.ndarray
            Current 20-dim drone state.
        target : array-like | callable
            (3,) target position, or a callable ``t -> (3,)`` for a moving
            target (``t`` is seconds since this behavior was reset).
        dash_speed : float
            Terminal dash speed (m/s); should exceed the cruise speed.
        hit_radius : float
            Distance (m) under which the strike is considered complete.
        timeout : float
            Safety cap (s) on the dash duration.
        decel_dist : float
            Distance (m) over which the dash speed is ramped down on final
            approach, so the strike converges on the target instead of
            overshooting through it.
        """
        super().reset(state)
        self.target_fn = target if callable(target) else (lambda _t, p=np.array(target, float): p)
        self.dash_speed = float(dash_speed)
        self.hit_radius = float(hit_radius)
        self.timeout = float(timeout)
        self.decel_dist = max(float(decel_dist), 1e-3)
        self.impact = False

    def step(self, state) -> Setpoint:
        tgt = np.asarray(self.target_fn(self.t), dtype=float)
        pos = np.array(state[POS], dtype=float)
        err = tgt - pos
        dist = float(np.linalg.norm(err))
        direction = err / dist if dist > 1e-6 else np.zeros(3)

        # Pin the position setpoint to the target (zero terminal error) and
        # push a strong velocity feed-forward for the max-acceleration dash,
        # ramped down within decel_dist so the drone converges instead of
        # blasting through the target.
        speed = self.dash_speed * min(1.0, dist / self.decel_dist)
        vel = speed * direction
        yaw = self._yaw_towards(pos, tgt)

        self.t += self.dt
        if dist < self.hit_radius:
            self.impact = True
            self._done = True
        elif self.t >= self.timeout:
            self._done = True
        return Setpoint(pos=tgt, rpy=np.array([0.0, 0.0, yaw]), vel=vel,
                        relax_safety=True)
