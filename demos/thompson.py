"""Posterior allocation of short optimizer continuations.

Capture and critical-point information are deliberately absent.  The
scheduler observes only the transformed loss after each allocated chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def transformed_loss(loss):
    """Map a nonnegative loss monotonically into [0,1]."""
    value = float(loss)
    if not np.isfinite(value):
        return 1.0
    if value <= 0.0:
        return 0.0
    if value > 1.0:
        return 1.0-1.0/(1.0+value)
    return value/(1.0+value)


@dataclass
class AllocationResult:
    trajectories: list
    allocations: np.ndarray
    observations: list
    choices: np.ndarray


def allocate(states, gradients, loss, posterior, rounds, chunk_steps=10):
    """Allocate ``rounds`` short continuations using posterior loss draws.

    Each arm receives one forced continuation before Thompson allocation,
    matching the historical SPONG experiment.  Thereafter the arm with the
    smallest posterior draw receives the next chunk.  No endpoint capture,
    basin label, or portrait geometry enters this decision.
    """
    n_arms = len(states)
    if n_arms == 0 or len(gradients) != n_arms:
        raise ValueError("states and gradients must have equal positive length")
    if rounds < n_arms:
        raise ValueError("rounds must allow one initial pull per arm")
    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")

    trajectories = [[np.asarray(state.z, dtype=float).copy()]
                    for state in states]
    observations = [[] for _ in states]
    allocations = np.zeros(n_arms, dtype=np.int64)
    choices = np.empty(rounds, dtype=np.int64)

    for round_index in range(rounds):
        arm = round_index if round_index < n_arms else int(
            np.argmin(posterior.draw_all()))
        state = states[arm]
        for _ in range(chunk_steps):
            if not np.all(np.isfinite(state.z)):
                break
            gradient = np.asarray(gradients[arm](*state.z), dtype=float)
            state.step(gradient)
            trajectories[arm].append(state.z.copy())
        observation = transformed_loss(loss(*state.z))
        posterior.update(arm, observation)
        observations[arm].append(observation)
        allocations[arm] += 1
        choices[round_index] = arm

    return AllocationResult(
        trajectories=[np.asarray(points) for points in trajectories],
        allocations=allocations,
        observations=observations,
        choices=choices)
