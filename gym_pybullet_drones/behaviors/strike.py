"""(4) Strike -- terminal dash onto a target.

RL command
    "Strike detected target T."

Abstraction
    A maximum-acceleration terminal dash that drives the position error between
    the drone and the target to zero. The position setpoint is pinned to the
    target, so the large standing position error commands a hard tilt straight
    along the line of sight. The heading is *held* at the value the drone enters
    the dash with: yawing to face the target would align the heading with the
    thrust direction and make the geometric attitude controller (which builds
    the body frame from heading x thrust) near-singular, stalling the dash.

    Two modes:

    * ``"guided"`` -- no velocity feed-forward. The position-error gain alone
      sets the tilt, so the dash naturally decelerates as it converges: a
      precise, soft-ish touch on the target.
    * ``"terminal"`` -- a committed ballistic dive. A strong velocity
      feed-forward along the line of sight is added on top of the pinned
      setpoint, so the drone accelerates to its tilt limit and plows through
      the target at high closing speed (a hard kill), sacrificing the gentle
      convergence of the guided mode.

    The safety filter is relaxed (``Setpoint.relax_safety``) so separation does
    not brake the dash. Impact is detected against the swept segment between
    consecutive positions, so a fast terminal pass cannot tunnel through the
    hit sphere between control steps.
"""
import numpy as np

from gym_pybullet_drones.behaviors.base_behavior import BaseBehavior, Setpoint, POS


class Strike(BaseBehavior):
    """Minimum-time terminal guidance onto a static or moving target."""

    name = "strike"

    def reset(self, state, target, mode="guided", dash_speed=3.0,
              hit_radius=0.15, timeout=10.0, **_):
        """Arm the dash toward ``target``.

        Parameters
        ----------
        state : np.ndarray
            Current 20-dim drone state.
        target : array-like | callable
            (3,) target position, or a callable ``t -> (3,)`` for a moving
            target (``t`` is seconds since this behavior was reset).
        mode : {"guided", "terminal"}
            ``"guided"`` drives a precise position dash (no feed-forward);
            ``"terminal"`` adds a strong velocity feed-forward for a committed
            high-speed ballistic dive through the target.
        dash_speed : float
            Velocity feed-forward magnitude (m/s) for ``"terminal"`` mode;
            ignored in ``"guided"`` mode.
        hit_radius : float
            Distance (m) under which the strike is considered complete.
        timeout : float
            Safety cap (s) on the dash duration.
        """
        super().reset(state)
        self.target_fn = target if callable(target) else (lambda _t, p=np.array(target, float): p)
        self.mode = str(mode)
        self.dash_speed = float(dash_speed)
        self.hit_radius = float(hit_radius)
        self.timeout = float(timeout)
        # Hold the entry heading (see module docstring) and seed the swept-
        # segment impact test with the start position.
        self.yaw = float(state[9])
        self.prev_pos = np.array(state[POS], dtype=float)
        self.impact = False
        # Fixed dash axis (entry position -> target), used by terminal mode to
        # detect the committed fly-through past the target.
        los0 = np.asarray(self.target_fn(0.0), dtype=float) - self.prev_pos
        n = float(np.linalg.norm(los0))
        self.dash_dir = los0 / n if n > 1e-6 else np.zeros(3)

    @staticmethod
    def _segment_dist(a, b, p):
        """Minimum distance from point ``p`` to the segment ``a``--``b``."""
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = np.clip(float(np.dot(p - a, ab)) / denom, 0.0, 1.0) if denom > 1e-12 else 0.0
        return float(np.linalg.norm(a + t * ab - p))

    def step(self, state) -> Setpoint:
        tgt = np.asarray(self.target_fn(self.t), dtype=float)
        pos = np.array(state[POS], dtype=float)
        err = tgt - pos
        dist = float(np.linalg.norm(err))
        direction = err / dist if dist > 1e-6 else np.zeros(3)

        # Pin the position setpoint to the target; the standing position error
        # drives a max-tilt dash. In terminal mode add a strong feed-forward
        # along the line of sight for a committed high-speed impact.
        vel = self.dash_speed * direction if self.mode == "terminal" else np.zeros(3)

        # Swept-segment impact test: catches a fast pass that would otherwise
        # tunnel through the hit sphere between control steps.
        if self._segment_dist(self.prev_pos, pos, tgt) < self.hit_radius:
            self.impact = True
            self._done = True
        self.prev_pos = pos.copy()

        # Terminal mode is a committed ballistic fly-through: once the drone
        # has passed the target along the dash axis, the dash is over. Finish
        # here so the pinned setpoint never pulls the overshot drone back to
        # the target (a clean miss flies on instead of looping around).
        if self.mode == "terminal" and float(np.dot(pos - tgt, self.dash_dir)) >= 0.0:
            self._done = True

        self.t += self.dt
        if self.t >= self.timeout:
            self._done = True
        return Setpoint(pos=tgt, rpy=np.array([0.0, 0.0, self.yaw]), vel=vel,
                        relax_safety=True)
