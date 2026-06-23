"""Cooperative task allocation: split one macro across N drones.

The :class:`SwarmCoordinator` receives a single high-level macro (e.g. "search
this disk") and uses these helpers to expand it into one sub-task per drone, so
the swarm shares the work:

    * Transit -> a formation slot per drone (they spread into a line)
    * Recon   -> a disk sub-region per drone (parallel bands or angular sectors)
    * Loiter  -> an evenly spaced orbit phase per drone
    * Strike  -> the same target for all (their distinct loiter bearings already
                 separate the terminal approach)

Geometry helpers are pure numpy; :func:`expand` turns a macro spec into the
per-drone ``(BehaviorType, params)`` list the FSMs consume.
"""
import numpy as np

from gym_pybullet_drones.behaviors.base_behavior import POS
from gym_pybullet_drones.utils.enums import BehaviorType


# ---------------------------------------------------------------------- #
# Geometry helpers
# ---------------------------------------------------------------------- #
def partition_bands(radius, n):
    """Split a disk into ``n`` equal-width horizontal bands (center-local y).

    Returns a list of ``("band", y_lo, y_hi)`` regions covering ``[-R, R]``.
    Equal width keeps the code simple; the middle bands hold slightly more
    area, so the phase barrier in the coordinator waits for the slowest drone.
    """
    edges = np.linspace(-radius, radius, n + 1)
    return [("band", float(edges[k]), float(edges[k + 1])) for k in range(n)]


def partition_sectors(n):
    """Split a disk into ``n`` equal angular wedges over ``[0, 2*pi)``.

    Returns a list of ``("sector", theta0, theta1)`` regions (radians).
    """
    edges = np.linspace(0.0, 2.0 * np.pi, n + 1)
    return [("sector", float(edges[k]), float(edges[k + 1])) for k in range(n)]


def orbit_phases(n):
    """Evenly spaced starting angles (rad) for ``n`` drones on one orbit."""
    return [2.0 * np.pi * k / n for k in range(n)]


def ring_slots(center, n, radius):
    """``n`` points evenly spaced on a horizontal ring around ``center``.

    Used to pre-position the swarm on the loiter orbit so that entering the
    loiter is jump-free.
    """
    center = np.asarray(center, dtype=float)
    return [center + np.array([radius * np.cos(2.0 * np.pi * k / n),
                               radius * np.sin(2.0 * np.pi * k / n), 0.0])
            for k in range(n)]


def formation_slots(target, n, spacing=0.4, axis=(0.0, 1.0, 0.0)):
    """Line-abreast formation slots centered on ``target``.

    Drones are placed symmetrically along ``axis`` so the group arrives spread
    out (which also keeps the simultaneous transit deconflicted).
    """
    target = np.asarray(target, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    return [target + (k - (n - 1) / 2.0) * spacing * axis for k in range(n)]


# ---------------------------------------------------------------------- #
# Macro -> per-drone expansion
# ---------------------------------------------------------------------- #
def expand(macro, num_drones, states):
    """Expand one cooperative macro into ``num_drones`` per-drone sub-tasks.

    Parameters
    ----------
    macro : dict
        ``{"type": BehaviorType, "params": {...}, "mode": str | None}``.
        ``mode`` selects the allocation scheme (e.g. ``"band"``/``"sector"``
        for Recon, ``"formation"`` for Transit).
    num_drones : int
        Number of drones to split the work across.
    states : np.ndarray
        (N, 20) current states; available for state-dependent allocation.

    Returns
    -------
    list[tuple[BehaviorType, dict]]
        One ``(behavior_type, params)`` command per drone.
    """
    bt = macro["type"]
    base = dict(macro.get("params", {}))
    mode = macro.get("mode")
    n = num_drones

    if bt == BehaviorType.TRANSIT:
        if mode == "formation":
            axis = base.get("axis", (0.0, 1.0, 0.0))
            slots = formation_slots(base["target"], n,
                                    spacing=base.get("spacing", 0.4), axis=axis)
            # Assign slots to drones in their current along-axis order so the
            # transit paths stay parallel and do not cross (deconfliction).
            targets = _assign_no_cross(slots, states, axis)
            params = _without(_without(base, "spacing"), "axis")
            return [(bt, {**params, "target": targets[k]}) for k in range(n)]
        if mode == "ring":
            # Pre-position the swarm on a ring (the upcoming loiter orbit) so
            # the loiter is entered without a jump. Assign each drone the ring
            # slot nearest its current bearing to avoid crossing paths. Because
            # surrounding a target from a one-sided approach forces some paths
            # to cross, give each drone a distinct altitude layer for this leg
            # so crossings are vertically deconflicted; the loiter then pulls
            # them back to a common altitude (a gentle vertical move).
            center = np.asarray(base["center"], dtype=float)
            layer_gap = base.get("layer_gap", 0.45)
            min_alt = base.get("min_alt", 0.15)
            # Shift the ring center up so the lowest layer stays above min_alt.
            max_neg_layer = ((n - 1) / 2.0) * layer_gap
            if center[2] - max_neg_layer < min_alt:
                center = center.copy()
                center[2] = min_alt + max_neg_layer
            slots = ring_slots(center, n, base["radius"])
            targets = _assign_ring(slots, states, center)
            params = _without(_without(_without(_without(base, "center"), "radius"),
                              "layer_gap"), "min_alt")
            out = []
            for k in range(n):
                layer = (k - (n - 1) / 2.0) * layer_gap
                out.append((bt, {**params,
                                 "target": targets[k] + np.array([0.0, 0.0, layer])}))
            return out
        return [(bt, base) for _ in range(n)]

    if bt == BehaviorType.RECON:
        cz = float(base["center"][2])
        gap = base.get("layer_gap", 0.3)
        params = _without(base, "layer_gap")
        if mode == "spiral":
            # Each drone flies a full Archimedean spiral rotated by 2π*k/n so
            # the swarm fans out from different starting angles simultaneously.
            # Altitude layers keep the interlaced paths vertically deconflicted.
            return [(bt, {**params, "pattern": "spiral",
                          "phase_offset": 2.0 * np.pi * k / n,
                          "altitude": cz + (k - (n - 1) / 2.0) * gap})
                    for k in range(n)]
        else:  # lawnmower: square spiral with phase offsets
            return [(bt, {**params, "pattern": "lawnmower",
                          "phase_offset": 2.0 * np.pi * k / n,
                          "altitude": cz + (k - (n - 1) / 2.0) * gap})
                    for k in range(n)]

    if bt == BehaviorType.LOITER:
        # Each drone enters the orbit at its *current* bearing (no jump). The
        # swarm is spread evenly by the preceding "ring" transit, so leaving
        # phase_offset unset keeps that spacing while avoiding a teleport.
        return [(bt, base) for _ in range(n)]

    if bt == BehaviorType.STRIKE:
        # Single-drone strike: the closest drone dashes onto the target;
        # the rest hold in place. Avoids the separation filter blocking
        # multiple simultaneous convergences.
        tgt = base["target"]
        params = _without(base, "ring")
        center = np.asarray(tgt(0.0) if callable(tgt) else tgt, dtype=float)
        dists = [np.linalg.norm(np.array(states[k][POS]) - center) for k in range(n)]
        striker = int(np.argmin(dists))
        return [(bt, {**params, "target": tgt}) if k == striker
                else (BehaviorType.IDLE, {})
                for k in range(n)]

    # IDLE or anything else: hold.
    return [(BehaviorType.IDLE, {}) for _ in range(n)]


def _assign_no_cross(slots, states, axis):
    """Match drones to formation slots by along-axis order (no crossing paths).

    Returns a per-drone list of target slots: the drone currently furthest
    along ``axis`` is sent to the slot furthest along ``axis``, etc.
    """
    n = len(slots)
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    drone_proj = np.array([np.dot(states[k][0:3], axis) for k in range(n)])
    slot_proj = np.array([np.dot(s, axis) for s in slots])
    drone_order = np.argsort(drone_proj)   # drone indices, low -> high
    slot_order = np.argsort(slot_proj)     # slot indices, low -> high
    targets = [None] * n
    for rank in range(n):
        targets[drone_order[rank]] = slots[slot_order[rank]]
    return targets


def _bearing_match(item_bearings, drone_bearings):
    """Permutation assigning items to drones in matching bearing order.

    Returns a list ``assign`` where ``assign[drone] = item_index``: the drone
    with the smallest bearing gets the item with the smallest bearing, etc., so
    that radial paths do not cross.
    """
    n = len(item_bearings)
    item_order = np.argsort(item_bearings)
    drone_order = np.argsort(drone_bearings)
    assign = [0] * n
    for rank in range(n):
        assign[drone_order[rank]] = int(item_order[rank])
    return assign


def _drone_bearings(states, center):
    """Bearing (rad) of each drone around ``center``."""
    return [np.arctan2(states[k][0:3][1] - center[1],
                       states[k][0:3][0] - center[0]) % (2.0 * np.pi)
            for k in range(len(states))]


def _assign_ring(slots, states, center):
    """Match drones to ring slots (absolute points) by current bearing."""
    center = np.asarray(center, dtype=float)
    item_b = [np.arctan2(s[1] - center[1], s[0] - center[0]) % (2.0 * np.pi)
              for s in slots]
    assign = _bearing_match(item_b, _drone_bearings(states, center))
    return [slots[assign[k]] for k in range(len(slots))]


def _assign_by_bearing(offsets, states, center):
    """Match drones to ring offsets (vectors) by current bearing."""
    center = np.asarray(center, dtype=float)
    item_b = [np.arctan2(o[1], o[0]) % (2.0 * np.pi) for o in offsets]
    assign = _bearing_match(item_b, _drone_bearings(states, center))
    return [offsets[assign[k]] for k in range(len(offsets))]


def _offset_target(target, offset):
    """Shift a target (static array or ``t -> (3,)`` callable) by ``offset``."""
    if callable(target):
        return lambda t, _f=target, _o=offset: np.asarray(_f(t), dtype=float) + _o
    return np.asarray(target, dtype=float) + offset


def _without(d, key):
    """Return a shallow copy of dict ``d`` without ``key``."""
    return {k: v for k, v in d.items() if k != key}
