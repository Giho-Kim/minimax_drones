"""Separation safety filter for the swarm.

A reactive layer inserted between the behaviors and the low-level PID. After
each drone's behavior emits a desired :class:`Setpoint`, this filter looks at
the *actual* pairwise distances and nudges the setpoints apart with an
artificial-potential-field repulsion, keeping at least ``d_min`` between
drones.

The push is added to both the position setpoint (so the PID actually tracks the
separation) and the velocity feed-forward (so it reacts promptly). A drone in
the Strike behavior has its safety relaxed -- it pushes others away far less --
modeling the doctrine that terminal attack constraints are partially lifted;
collision avoidance there is instead handled geometrically by giving strikers
distinct approach bearings upstream (see allocation).
"""
import numpy as np

from gym_pybullet_drones.behaviors.base_behavior import POS


class SeparationFilter:
    """Artificial-potential-field minimum-separation filter."""

    def __init__(self, d_min=0.4, gain=1.5, max_push=0.3, vel_gain=0.5,
                 relax_factor=0.15, horizontal_only=True, z_clear=0.25):
        """Parameters
        ----------
        d_min : float
            Minimum desired separation (m); repulsion is zero beyond it.
        gain : float
            Strength of the positional repulsion.
        max_push : float
            Cap (m) on the positional nudge applied to any one drone.
        vel_gain : float
            How much of the positional push is mirrored into the velocity
            feed-forward.
        relax_factor : float
            Multiplier (<1) on the repulsion *produced by* a drone whose
            setpoint has ``relax_safety`` set (i.e. a striker).
        horizontal_only : bool
            If True, repulsion acts only in the XY plane. Vertical shoves tend
            to drive these small drones into the floor and separation is almost
            always achievable horizontally, so this defaults on.
        z_clear : float
            Vertical clearance (m). Pairs separated by more than this in
            altitude are treated as already deconflicted and not pushed -- this
            lets altitude-layered drones (e.g. adjacent recon sub-regions or
            crossing transit legs) cover shared boundaries without the filter
            shoving them around.
        """
        self.d_min = float(d_min)
        self.gain = float(gain)
        self.max_push = float(max_push)
        self.vel_gain = float(vel_gain)
        self.relax_factor = float(relax_factor)
        self.horizontal_only = bool(horizontal_only)
        self.z_clear = float(z_clear)

    def apply(self, setpoints, states):
        """Deconflict ``setpoints`` in place using the drones' actual states.

        Parameters
        ----------
        setpoints : list[Setpoint]
            One desired setpoint per drone (mutated and returned).
        states : np.ndarray
            (N, 20) current drone states.

        Returns
        -------
        list[Setpoint]
            The same list, with positions/velocities nudged for separation.
        """
        n = len(setpoints)
        if n < 2:
            return setpoints

        pos = np.array([states[i][POS] for i in range(n)], dtype=float)
        # Per-drone weight on the repulsion it emits (strikers push weakly).
        emit = np.array([self.relax_factor if setpoints[j].relax_safety else 1.0
                         for j in range(n)])

        for i in range(n):
            push = np.zeros(3)
            for j in range(n):
                if i == j:
                    continue
                d = pos[i] - pos[j]
                if abs(d[2]) > self.z_clear:        # vertically deconflicted
                    continue
                dist = np.linalg.norm(d)            # full 3D conflict test
                if not (1e-6 < dist < self.d_min):
                    continue
                # Repulse horizontally only (vertical shoves drive these small
                # drones into the floor). Drones separated mostly in altitude
                # have a small horizontal component and so are barely pushed.
                dh = d.copy()
                if self.horizontal_only:
                    dh[2] = 0.0
                nh = np.linalg.norm(dh)
                if nh > 1e-6:
                    push += emit[j] * (dh / nh) * (self.d_min - dist)
            push *= self.gain
            norm = np.linalg.norm(push)
            if norm > self.max_push:
                push *= self.max_push / norm
            setpoints[i].pos = setpoints[i].pos + push
            setpoints[i].vel = setpoints[i].vel + self.vel_gain * push
        return setpoints
