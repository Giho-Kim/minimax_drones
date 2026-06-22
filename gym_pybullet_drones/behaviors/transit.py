"""(1) Transit -- point-to-point move along the line of sight.

RL command
    "Move to coordinate (x, y, z)."

Abstraction
    Generate the straight 3-D line of sight from the current position to the
    goal and follow a point-to-point trajectory that respects a maximum speed
    and acceleration. The along-track motion uses a trapezoidal velocity
    profile (accelerate -> cruise -> decelerate); if the distance is too short
    to reach cruise speed the profile degenerates to a triangle.
"""
import numpy as np

from gym_pybullet_drones.behaviors.base_behavior import BaseBehavior, Setpoint, POS


class Transit(BaseBehavior):
    """Straight-line point-to-point transit with a trapezoidal speed profile."""

    name = "transit"

    def reset(self, state, target, v_max=0.6, a_max=1.0, yaw=None,
              reach_tol=0.12, settle_max=4.0, **_):
        """Plan the trapezoidal profile from the current position to ``target``.

        Parameters
        ----------
        state : np.ndarray
            Current 20-dim drone state; ``state[POS]`` is the start point.
        target : array-like
            (3,) goal position in the world frame.
        v_max : float
            Cruise speed (m/s).
        a_max : float
            Acceleration / deceleration magnitude (m/s^2).
        yaw : float | None
            Desired yaw. If ``None`` the drone faces its direction of travel.
        reach_tol : float
            Distance (m) under which the goal is considered physically reached.
        settle_max : float
            Extra time (s) after the planned trajectory ends to wait for the
            drone to actually arrive before giving up and reporting done. This
            keeps a swarm phase barrier from advancing while drones still lag
            (e.g. still climbing), which would start the next behavior from a
            bad state.
        """
        super().reset(state)
        self.start = np.array(state[POS], dtype=float)
        self.goal = np.array(target, dtype=float)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.reach_tol = float(reach_tol)
        self.settle_max = float(settle_max)

        delta = self.goal - self.start
        self.dist = float(np.linalg.norm(delta))
        self.dir = delta / self.dist if self.dist > 1e-9 else np.zeros(3)

        # Face the direction of travel unless an explicit yaw was requested.
        self.yaw = self._yaw_towards(self.start, self.goal) if yaw is None else float(yaw)

        # --- Plan the along-track trapezoidal profile -------------------- #
        d_acc = 0.5 * self.v_max ** 2 / self.a_max   # distance to reach v_max
        if 2.0 * d_acc >= self.dist:
            # Triangular profile: never reach cruise speed.
            self.t_acc = np.sqrt(self.dist / self.a_max) if self.a_max > 0 else 0.0
            self.v_peak = self.a_max * self.t_acc
            self.t_cruise = 0.0
        else:
            self.t_acc = self.v_max / self.a_max
            self.v_peak = self.v_max
            d_cruise = self.dist - 2.0 * d_acc
            self.t_cruise = d_cruise / self.v_max
        self.T = 2.0 * self.t_acc + self.t_cruise

        if self.dist <= self.reach_tol:
            self._done = True

    def _arclength(self, t):
        """Return (s, v): along-track distance and speed at elapsed time ``t``."""
        if t <= self.t_acc:                                   # accelerating
            s = 0.5 * self.a_max * t ** 2
            v = self.a_max * t
        elif t <= self.t_acc + self.t_cruise:                 # cruising
            s = 0.5 * self.a_max * self.t_acc ** 2 + self.v_peak * (t - self.t_acc)
            v = self.v_peak
        elif t <= self.T:                                     # decelerating
            td = t - self.t_acc - self.t_cruise
            s = (0.5 * self.a_max * self.t_acc ** 2
                 + self.v_peak * self.t_cruise
                 + self.v_peak * td - 0.5 * self.a_max * td ** 2)
            v = self.v_peak - self.a_max * td
        else:                                                 # arrived
            s, v = self.dist, 0.0
        return min(s, self.dist), max(v, 0.0)

    def step(self, state) -> Setpoint:
        s, v = self._arclength(self.t)
        pos = self.start + s * self.dir
        vel = v * self.dir
        self.t += self.dt
        # The planned trajectory is finished, but only report done once the
        # drone has physically arrived (or the settle window expires), holding
        # at the goal in the meantime.
        if self.t >= self.T:
            arrived = np.linalg.norm(self.goal - np.array(state[POS])) < self.reach_tol
            if arrived or self.t >= self.T + self.settle_max:
                self._done = True
        return Setpoint(pos=pos, rpy=np.array([0.0, 0.0, self.yaw]), vel=vel)
