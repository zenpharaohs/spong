"""Posterior allocation of short optimizer continuations.

Capture and critical-point information are deliberately absent.  The
scheduler observes only the transformed loss after each allocated chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from demos import optimizers


def transformed_loss(loss):
    """Map a nonnegative loss monotonically into the closed unit interval."""
    value = float(loss)
    if not np.isfinite(value):
        return 1.0
    if value <= 0.0:
        return 0.0
    if value > 1.0:
        observation = 1.0-1.0/(1.0+value)
    else:
        observation = value/(1.0+value)
    return min(1.0, max(0.0, observation))


@dataclass
class AllocationResult:
    trajectories: list
    allocations: np.ndarray
    observations: list
    choices: np.ndarray
    executed_steps: np.ndarray
    termination_reasons: tuple
    allocation_losses: np.ndarray
    allocation_steps: np.ndarray

    @property
    def terminated(self):
        return np.asarray(
            [reason is not None and reason != "zero_loss"
             for reason in self.termination_reasons],
            dtype=bool)


def _raw_loss(loss, z):
    """Evaluate exact loss without leaking numerical warnings."""
    if not np.all(np.isfinite(z)):
        return float("inf")
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            value = float(loss(*z))
    except ArithmeticError:
        return float("inf")
    return value if np.isfinite(value) else float("inf")


def _observe(loss, z):
    """Evaluate the posterior statistic without leaking overflow warnings."""
    return transformed_loss(_raw_loss(loss, z))


@dataclass
class ChunkRecord:
    points: list
    raw_loss: float
    reason: str | None


@dataclass
class ArmTape:
    """One optimizer stream, recorded by chunk for both allocation policies."""
    state: object
    gradient: object
    loss: object
    chunk_steps: int
    start: np.ndarray
    records: list
    terminal_reason: str | None = None

    @classmethod
    def create(cls, state, gradient, loss, chunk_steps):
        return cls(state, gradient, loss, chunk_steps,
                   np.asarray(state.z, dtype=float).copy(), [])

    def extend_to(self, pulls, should_stop=None):
        while len(self.records) < pulls and self.terminal_reason is None:
            points = []
            reason = None
            for _ in range(self.chunk_steps):
                if should_stop is not None and should_stop():
                    raise TimeoutError("allocation stopped by interactive time limit")
                z, failure = optimizers.checked_step(self.state, self.gradient)
                if failure is not None:
                    reason = failure
                    break
                points.append(z.copy())
            raw_loss = _raw_loss(self.loss, self.state.z)
            observation = transformed_loss(raw_loss)
            if reason is None and not np.isfinite(raw_loss):
                reason = "nonfinite_loss"
            elif reason is None and observation >= 1.0:
                reason = "loss_transform_saturated"
            self.records.append(ChunkRecord(points, raw_loss, reason))
            # Zero is observed by the policy before it stops globally; it is
            # not a numerical failure and must not be fed to deactivate().
            self.terminal_reason = ("zero_loss" if reason is None
                                    and observation == 0.0 else reason)

    def pull(self, ordinal, should_stop=None):
        self.extend_to(ordinal + 1, should_stop=should_stop)
        return self.records[ordinal]


class TrajectoryCache:
    def __init__(self, states, gradients, loss, chunk_steps):
        self.arms = [ArmTape.create(state, gradient, loss, chunk_steps)
                     for state, gradient in zip(states, gradients)]

    def pull(self, arm, ordinal, should_stop=None):
        return self.arms[arm].pull(ordinal, should_stop=should_stop)


def allocate(states, gradients, loss, posterior, rounds, chunk_steps=10,
             should_stop=None, cache=None):
    """Allocate ``rounds`` short continuations using posterior loss draws.

    Each arm receives one forced continuation before Thompson allocation,
    matching the historical SPONG experiment.  Thereafter the arm with the
    smallest posterior draw receives the next chunk.  No endpoint capture,
    basin label, or portrait geometry enters this decision.

    Exact descent-bandit streams are held at ``(N_i Z_i, N_i)``, where
    ``Z_i`` is the arm's current observation; they do not accumulate the
    obsolete losses encountered earlier on the same trajectory.  A simple
    test posterior may omit ``set_observation`` and retain the historical
    ``update`` protocol.

    ``cache`` optionally supplies recorded chunks from independent arm tapes.
    Both policies can then consume identical optimizer evidence, with a tape
    extending its state only when a policy requests a new chunk.

    ``should_stop`` is an optional cooperative interruption hook for
    interactive consumers.  A stopped comparison is discarded rather than
    returned with unequal policy budgets.

    """
    n_arms = len(states)
    if n_arms == 0 or len(gradients) != n_arms:
        raise ValueError("states and gradients must have equal positive length")
    if rounds < n_arms:
        raise ValueError("rounds must allow one initial pull per arm")
    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")

    trajectories = [[(cache.arms[i].start if cache is not None
                      else np.asarray(state.z, dtype=float)).copy()]
                    for i, state in enumerate(states)]
    observations = [[] for _ in states]
    allocations = np.zeros(n_arms, dtype=np.int64)
    choices = np.empty(rounds, dtype=np.int64)
    executed_steps = np.zeros(n_arms, dtype=np.int64)
    termination_reasons = [None] * n_arms
    allocation_losses = np.empty(rounds, dtype=float)
    allocation_steps = np.zeros(rounds, dtype=np.int64)
    completed_rounds = 0

    for round_index in range(rounds):
        if should_stop is not None and should_stop():
            raise TimeoutError("allocation stopped by interactive time limit")
        if round_index < n_arms:
            arm = round_index
        else:
            if all(reason is not None for reason in termination_reasons):
                break
            draws = np.asarray(posterior.draw_all(), dtype=float)
            terminal_mask = np.asarray(
                [reason is not None for reason in termination_reasons])
            draws[terminal_mask] = np.inf
            arm = int(np.argmin(draws))
        steps_before = int(executed_steps[arm])
        if termination_reasons[arm] is None:
            if cache is not None:
                record = cache.pull(arm, int(allocations[arm]),
                                    should_stop=should_stop)
                trajectories[arm].extend(record.points)
                executed_steps[arm] += len(record.points)
                termination_reasons[arm] = record.reason
            else:
                state = states[arm]
                for _ in range(chunk_steps):
                    if should_stop is not None and should_stop():
                        raise TimeoutError(
                            "allocation stopped by interactive time limit")
                    z, failure = optimizers.checked_step(
                        state, gradients[arm])
                    if failure is not None:
                        termination_reasons[arm] = failure
                        break
                    trajectories[arm].append(z.copy())
                    executed_steps[arm] += 1
        # Once numerical divergence occurs, preserve the last finite point
        # and remove the arm from future selection.  The sentinel observation
        # remains in the result for diagnosis; it is not fed to the sampler.
        raw_loss = (record.raw_loss if cache is not None
                    else _raw_loss(loss, state.z))
        observation = transformed_loss(raw_loss)
        if termination_reasons[arm] is None and not np.isfinite(raw_loss):
            termination_reasons[arm] = "nonfinite_loss"
        elif termination_reasons[arm] is None and observation >= 1.0:
            termination_reasons[arm] = "loss_transform_saturated"

        terminal = termination_reasons[arm] is not None
        if terminal:
            observation = 1.0
            deactivate = getattr(posterior, "deactivate", None)
            if deactivate is not None:
                deactivate(arm)
        else:
            selections = int(allocations[arm]) + 1
            set_observation = getattr(posterior, "set_observation", None)
            if set_observation is None:
                posterior.update(arm, observation)
            else:
                set_observation(arm, observation, selections)
        observations[arm].append(observation)
        allocations[arm] += 1
        choices[round_index] = arm
        allocation_losses[round_index] = raw_loss
        allocation_steps[round_index] = int(executed_steps[arm])-steps_before
        completed_rounds = round_index + 1

        # Zero is the resolved lower endpoint of the closed family.  Once an
        # optimizer has attained it, spending more allocation rounds cannot
        # improve the objective.
        if not terminal and observation == 0.0:
            termination_reasons[arm] = "zero_loss"
            break

    return AllocationResult(
        trajectories=[np.asarray(points) for points in trajectories],
        allocations=allocations,
        observations=observations,
        choices=choices[:completed_rounds],
        executed_steps=executed_steps,
        termination_reasons=tuple(termination_reasons),
        allocation_losses=allocation_losses[:completed_rounds],
        allocation_steps=allocation_steps[:completed_rounds])
