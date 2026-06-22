"""(2) Recon / Search -- cover a disk around a center point.

RL command
    "Search the area within radius R around coordinate (x, y, z)."

Abstraction
    The AI does *not* steer the drone every second. Instead the drone flies to
    the commanded center and then autonomously unrolls a built-in search
    pattern (lawnmower or spiral) at a fixed altitude, maximizing the sensor
    coverage of the disk of radius ``R``. The pattern is pre-computed as a
    polyline of waypoints; the drone follows it at a constant cruise speed
    using arc-length parameterization for smooth, non-stop motion.
"""
import numpy as np

from gym_pybullet_drones.behaviors.base_behavior import BaseBehavior, Setpoint, POS


class Recon(BaseBehavior):
    """Coverage of a disk using a lawnmower or spiral pattern."""

    name = "recon"

    def reset(self, state, center, radius, pattern="lawnmower", swath=0.5,
              speed=0.6, altitude=None, region=None, **_):
        """Plan a coverage polyline over the disk and start following it.

        Parameters
        ----------
        state : np.ndarray
            Current 20-dim drone state; the path starts at ``state[POS]``.
        center : array-like
            (3,) center of the search area. ``center[2]`` is the search altitude
            unless ``altitude`` is given.
        radius : float
            Radius (m) of the area to cover.
        pattern : {"lawnmower", "spiral"}
            Coverage pattern to deploy.
        swath : float
            Sensor footprint width (m); the spacing between adjacent passes.
        speed : float
            Cruise speed (m/s) along the pattern.
        altitude : float | None
            Override search altitude; defaults to ``center[2]``.
        region : tuple | None
            Sub-region of the disk to cover, used to split coverage across a
            swarm. ``None`` covers the whole disk. Otherwise one of:

            * ``("band", y_lo, y_hi)`` -- a horizontal strip (center-relative
              local y); pairs naturally with the lawnmower pattern.
            * ``("sector", theta0, theta1)`` -- an angular wedge in radians,
              ``[0, 2*pi)``; pairs naturally with the spiral pattern.
        """
        super().reset(state)
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.swath = max(float(swath), 1e-3)
        self.speed = float(speed)
        self.alt = float(center[2] if altitude is None else altitude)
        self.region = region

        # Scan-line y-limits (a "band" region restricts them; else full disk).
        if region is not None and region[0] == "band":
            self.y_lo = max(-self.radius, float(region[1]))
            self.y_hi = min(self.radius, float(region[2]))
        else:
            self.y_lo, self.y_hi = -self.radius, self.radius

        if region is not None and region[0] == "sector":
            # A sector sub-region gets a dedicated continuous wedge sweep,
            # regardless of the requested pattern (filtering a spiral to a wedge
            # leaves jumpy, disconnected arcs).
            pts = self._sector_pattern(float(region[1]), float(region[2]))
        elif pattern == "spiral":
            pts = self._spiral_pattern()
        else:
            pts = self._lawnmower_pattern()

        # Clip to the assigned region (handles "sector"; "band" already limited).
        pts = [pt for pt in pts if self._in_region(pt)]

        # Prepend the current position so the drone smoothly flies in.
        wps = [np.array(state[POS], dtype=float)] + pts
        self.waypoints = np.array(wps, dtype=float)

        # Cumulative arc length for parameterized following.
        seg = np.diff(self.waypoints, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.cum_len = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.total_len = float(self.cum_len[-1])
        self.s = 0.0
        if self.total_len < 1e-6:
            self._done = True

    # ------------------------------------------------------------------ #
    # Pattern generators (in the disk's local XY frame, then offset)
    # ------------------------------------------------------------------ #
    def _in_region(self, pt):
        """Whether a world-frame point falls in the assigned sub-region."""
        if self.region is None:
            return True
        rel = np.asarray(pt) - self.center
        if self.region[0] == "band":
            return self.y_lo - 1e-9 <= rel[1] <= self.y_hi + 1e-9
        if self.region[0] == "sector":
            ang = np.arctan2(rel[1], rel[0]) % (2.0 * np.pi)
            return float(self.region[1]) <= ang < float(self.region[2])
        return True

    def _lawnmower_pattern(self):
        """Boustrophedon (back-and-forth) scan lines clipped to the disk."""
        R = self.radius
        pts = []
        ys = np.arange(self.y_lo, self.y_hi + 1e-9, self.swath)
        for k, y in enumerate(ys):
            half = np.sqrt(max(R ** 2 - y ** 2, 0.0))  # x-extent of the chord
            if half < 1e-6:
                continue
            xs = [-half, half] if k % 2 == 0 else [half, -half]
            for x in xs:
                pts.append(self.center + np.array([x, y, self.alt - self.center[2]]))
        return pts

    def _spiral_pattern(self):
        """Archimedean spiral expanding out to the disk radius."""
        R = self.radius
        b = self.swath / (2.0 * np.pi)          # radial growth per revolution
        theta_max = R / b if b > 0 else 0.0
        # Sample densely enough that segments are shorter than the swath.
        n = max(int(theta_max / (np.pi / 8)), 2)
        thetas = np.linspace(0.0, theta_max, n)
        pts = []
        for th in thetas:
            r = min(b * th, R)
            x, y = r * np.cos(th), r * np.sin(th)
            pts.append(self.center + np.array([x, y, self.alt - self.center[2]]))
        return pts

    def _sector_pattern(self, theta0, theta1):
        """Continuous wedge sweep: concentric arcs across ``[theta0, theta1]``.

        Sweeps arcs at increasing radius, alternating angular direction each
        ring (a boustrophedon in polar coordinates), so the path stays
        connected with no large radius jumps -- unlike filtering a spiral to
        the wedge.
        """
        R = self.radius
        span = theta1 - theta0
        z_off = self.alt - self.center[2]
        radii = np.arange(self.swath * 0.5, R + 1e-9, self.swath)
        pts = []
        for i, r in enumerate(radii):
            # Enough angular samples that arc segments stay shorter than swath.
            nseg = max(2, int(r * span / self.swath))
            angs = np.linspace(theta0, theta1, nseg)
            if i % 2 == 1:                      # alternate direction each ring
                angs = angs[::-1]
            for a in angs:
                pts.append(self.center + np.array([r * np.cos(a), r * np.sin(a), z_off]))
        return pts

    # ------------------------------------------------------------------ #
    def _point_at(self, s):
        """Interpolate position and unit travel direction at arc length ``s``."""
        s = np.clip(s, 0.0, self.total_len)
        idx = int(np.searchsorted(self.cum_len, s, side="right") - 1)
        idx = np.clip(idx, 0, len(self.seg_len) - 1)
        seg_len = self.seg_len[idx]
        frac = (s - self.cum_len[idx]) / seg_len if seg_len > 1e-9 else 0.0
        a, b = self.waypoints[idx], self.waypoints[idx + 1]
        pos = a + frac * (b - a)
        direction = (b - a) / seg_len if seg_len > 1e-9 else np.zeros(3)
        return pos, direction

    def step(self, state) -> Setpoint:
        pos, direction = self._point_at(self.s)
        vel = self.speed * direction
        yaw = self._yaw_towards(np.zeros(3), direction) if np.linalg.norm(direction[:2]) > 1e-6 \
            else float(state[9])
        self.s += self.speed * self.dt
        if self.s >= self.total_len:
            self._done = True
        return Setpoint(pos=pos, rpy=np.array([0.0, 0.0, yaw]), vel=vel)
