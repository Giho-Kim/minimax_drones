"""Cooperative multi-drone coordination on top of the macro behaviors.

A :class:`SwarmCoordinator` sequences one shared mission across N drones,
splitting each macro (Transit / Recon / Loiter / Strike) among them with the
:mod:`allocation` helpers and keeping them apart with a
:class:`SeparationFilter`. The per-drone behavior FSMs in
:mod:`gym_pybullet_drones.behaviors` are reused unchanged.
"""
from gym_pybullet_drones.swarm.coordinator import SwarmCoordinator
from gym_pybullet_drones.swarm.deconfliction import SeparationFilter
from gym_pybullet_drones.swarm import allocation

__all__ = ["SwarmCoordinator", "SeparationFilter", "allocation"]
