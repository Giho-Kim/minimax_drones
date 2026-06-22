"""(3) Loiter / Track -- circular orbit that keeps a target in view.

RL command
    "Surveil target T (or a suspected area)."

Abstraction
    Once over the target the drone computes the orbit radius and altitude that
    keep the target centered in the sensor field of view, then flies a circular
    loiter. The orbit radius is derived from the standoff altitude and the
    desired sensor depression angle (``R = h / tan(depression)``), modeling a
    gimbal that stares at the orbit center. If the target moves, the orbit
    center moves with it.
"""
import numpy as np

from gym_pybullet_drones.behaviors.base_behavior import BaseBehavior, Setpoint, POS


class Loiter(BaseBehavior):
    """Circular loitering around a static or moving target."""

    name = "loiter"

    def reset(self, state, target, standoff_alt=0.8, depression_deg=45.0,
              orbit_speed=0.6, duration=8.0, radius=None, phase_offset=None, **_):
        """Configure the orbit around ``target``.

        Parameters
        ----------
        state : np.ndarray
            Current 20-dim drone state.
        target : array-like | callable
            (3,) target position, or a callable ``t -> (3,)`` for a moving
            target (``t`` is seconds since this behavior was reset).
        standoff_alt : float
            Height (m) to hold above the target.
        depression_deg : float
            Sensor depression angle; sets the orbit radius via
            ``R = h / tan(depression)``.
        orbit_speed : float
            Tangential speed (m/s) along the circle.
        duration : float
            How long (s) to loiter before reporting done.
        radius : float | None
            Explicit orbit radius (m); overrides the FOV-derived value.
        phase_offset : float | None
            Fixed starting angle (rad) on the orbit. Used to spread a swarm
            evenly around the target (drone k gets ``2*pi*k/N``). If ``None``
            the drone enters the orbit at its current bearing from the target.
        """
        super().reset(state)
        self.target_fn = target if callable(target) else (lambda _t, p=np.array(target, float): p)
        self.standoff_alt = float(standoff_alt)
        self.orbit_speed = float(orbit_speed)
        self.duration = float(duration)

        if radius is not None:
            self.radius = float(radius)
        else:
            dep = np.radians(np.clip(depression_deg, 1.0, 89.0))
            self.radius = self.standoff_alt / np.tan(dep)
        self.radius = max(self.radius, 1e-3)

        # Angular rate to achieve the requested tangential speed.
        self.omega = self.orbit_speed / self.radius

        # Spread a swarm with an explicit phase, else enter the orbit at the
        # current bearing from the target so the transition is smooth.
        if phase_offset is not None:
            self.phase = float(phase_offset)
        else:
            c = self.target_fn(0.0)
            rel = np.array(state[POS])[:2] - c[:2]
            self.phase = float(np.arctan2(rel[1], rel[0])) if np.linalg.norm(rel) > 1e-6 else 0.0

    def step(self, state) -> Setpoint:
        center = np.asarray(self.target_fn(self.t), dtype=float)
        a = self.phase + self.omega * self.t

        pos = center + np.array([self.radius * np.cos(a),
                                 self.radius * np.sin(a),
                                 self.standoff_alt])
        # Tangential velocity around the circle (target drift is tracked via
        # the moving center; we keep the feed-forward to the orbital motion).
        vel = self.orbit_speed * np.array([-np.sin(a), np.cos(a), 0.0])
        # Keep the sensor pointed inward at the target.
        yaw = self._yaw_towards(pos, center)

        self.t += self.dt
        if self.t >= self.duration:
            self._done = True
        return Setpoint(pos=pos, rpy=np.array([0.0, 0.0, yaw]), vel=vel)
