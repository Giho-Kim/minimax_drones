"""SwarmCoordinator: drive N drones through one shared, cooperative mission.

The coordinator is the multi-drone analogue of a single :class:`BehaviorManager`
and the layer an RL policy would eventually replace. It holds one FSM per drone
plus a separation filter, and sequences a mission of high-level macros with a
*phase barrier*: every drone must finish the current macro before the swarm
moves on to the next. Each macro is split across the drones by the allocation
helpers, so the swarm cooperates on one task at a time.

    mission (list of macros)
        |
        v
    SwarmCoordinator
        |-- allocation.expand(macro) ---> per-drone (behavior, params)
        |-- BehaviorManager[k].step(state_k) ---> setpoint_k
        |-- SeparationFilter.apply(setpoints, states)
        v
    N setpoints  -->  DSLPIDControl[k]  -->  RPMs
"""
import numpy as np

from gym_pybullet_drones.utils.enums import BehaviorType
from gym_pybullet_drones.behaviors.fsm import BehaviorManager
from gym_pybullet_drones.swarm import allocation
from gym_pybullet_drones.swarm.deconfliction import SeparationFilter


class SwarmCoordinator:
    """Sequences a cooperative mission across N per-drone FSMs."""

    def __init__(self, num_drones, ctrl_freq, mission=None,
                 separation_filter=None):
        """Parameters
        ----------
        num_drones : int
            Number of drones in the swarm.
        ctrl_freq : int
            Control frequency (Hz) handed to each behavior FSM.
        mission : list[dict] | None
            Ordered list of macro specs (see :func:`allocation.expand`). Can
            also be set later with :meth:`set_mission`.
        separation_filter : SeparationFilter | None
            Deconfliction filter; a default one is created if omitted.
        """
        self.N = num_drones
        self.managers = [BehaviorManager(ctrl_freq) for _ in range(num_drones)]
        self.filter = separation_filter if separation_filter is not None \
            else SeparationFilter()
        self.mission = list(mission) if mission else []
        self.phase_idx = 0          # index of the next macro to assign
        self._started = False
        # Diagnostics: which macro the swarm is currently executing.
        self.current_macro = None

    def set_mission(self, mission):
        """Replace the mission (list of macro specs). Returns ``self``."""
        self.mission = list(mission)
        self.phase_idx = 0
        self._started = False
        self.current_macro = None
        return self

    # ------------------------------------------------------------------ #
    def _phase_complete(self):
        """True once every drone has finished the current macro."""
        return all(m.finished for m in self.managers)

    def _assign_phase(self, macro, states):
        """Split ``macro`` across drones and enqueue each FSM's sub-task."""
        per_drone = allocation.expand(macro, self.N, states)
        for k, (btype, params) in enumerate(per_drone):
            self.managers[k].command(btype, **params)
        self.current_macro = macro

    def step(self, obs):
        """Advance the swarm by one control step.

        Parameters
        ----------
        obs : np.ndarray
            (N, 20) current states from the environment.

        Returns
        -------
        list[Setpoint]
            One deconflicted setpoint per drone, ready for the PID controllers.
        """
        # Phase barrier: assign the first macro, or the next one once every
        # drone has finished the current macro.
        if self.phase_idx < len(self.mission) and \
                (not self._started or self._phase_complete()):
            self._assign_phase(self.mission[self.phase_idx], obs)
            self.phase_idx += 1
            self._started = True

        setpoints = [self.managers[k].step(obs[k]) for k in range(self.N)]
        return self.filter.apply(setpoints, obs)

    @property
    def behavior_types(self):
        """List of each drone's currently active :class:`BehaviorType`."""
        return [m.current_type for m in self.managers]

    @property
    def finished(self):
        """True once all macros are assigned and every drone has settled."""
        return self._started and self.phase_idx >= len(self.mission) \
            and self._phase_complete()
