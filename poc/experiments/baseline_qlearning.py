"""Tabular Q-learning baseline for comparison against the concept-table
ablation ladder.

The internal baselines (random, memory-only) establish that this
architecture's own components each contribute measurably. They do not, on
their own, establish that the approach is competitive with a standard
reinforcement-learning alternative. This module adds a classic tabular
Q-learning agent -- a lookup table updated by the Bellman equation, with
no function approximation and no neural network -- as that external
reference point.

Scope: evaluated only on the single-feature entity-discovery environment,
not extended to the conjunctive-rule, planning, or curiosity environments,
which would require a substantially larger state space and training
budget to converge fairly.
"""

import random
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gridworld import fresh_training_world, MOVES, ACTIONS, RED
from stats_utils import significant_improvement, cohens_d

from experiments.exp0_entity_discovery import (
    train as train_generic, transfer_test, TRAIN_EPISODES, N_SEEDS
)
from agent.agent import FullAgent, RandomAgent, MemoryOnlyAgent

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.15


class QLearningAgent:
    """Standard epsilon-greedy tabular Q-learning agent. State is defined
    as (position, feature at current cell). Hyperparameters (alpha, gamma,
    epsilon) are set to reasonable defaults without a tuning pass, matching
    the standard applied to this project's other hyperparameters.
    """

    def __init__(self):
        self.Q = {}
        self.last_state = None
        self.last_action = None

    def _state(self, obs):
        return (obs["pos"], obs["feature"])

    def _q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def act(self, obs):
        state = self._state(obs)
        if random.random() < EPSILON:
            action = random.choice(ACTIONS)
        else:
            qs = [(self._q(state, a), a) for a in ACTIONS]
            max_q = max(q for q, _ in qs)
            best_actions = [a for q, a in qs if q == max_q]
            action = random.choice(best_actions)
        self.last_state = state
        self.last_action = action
        return action

    def observe_result(self, obs, action, next_obs, delta_energy):
        state = self._state(obs)
        next_state = self._state(next_obs)
        max_next_q = max(self._q(next_state, a) for a in ACTIONS)
        old_q = self._q(state, action)
        new_q = old_q + ALPHA * (delta_energy + GAMMA * max_next_q - old_q)
        self.Q[(state, action)] = new_q


def train_qlearning(episodes=TRAIN_EPISODES):
    from experiments.exp0_entity_discovery import run_episode
    agent = QLearningAgent()
    survival = []
    for _ in range(episodes):
        world = fresh_training_world()
        _, steps = run_episode(agent, world)
        survival.append(steps)
    return agent, survival


def main():
    print(f"\n=== Tabular Q-Learning Baseline vs. Ablation Ladder ({N_SEEDS} seeds) ===\n")
    print("Scope: single-feature entity-discovery and transfer environment only.")
    print("See module docstring for the reasoning behind this scope limit.\n")

    results = {}
    for name, train_fn in [
        ("random", lambda: train_generic(RandomAgent)),
        ("memory_only", lambda: train_generic(MemoryOnlyAgent)),
        ("full (ours)", lambda: train_generic(FullAgent)),
        ("q_learning", train_qlearning),
    ]:
        survivals, transfers = [], []
        for seed in range(N_SEEDS):
            random.seed(seed)
            agent, survival = train_fn()
            survivals.append(sum(survival) / len(survival))
            rate, _ = transfer_test(agent)
            transfers.append(rate)
        results[name] = {"survival": survivals, "transfer": transfers}

    print(f"{'agent':<16}{'survival (mean/std)':<22}{'transfer_rate (mean/std)'}")
    print("-" * 62)
    for name, r in results.items():
        s_m, s_s = round(statistics.mean(r["survival"]), 1), round(statistics.stdev(r["survival"]), 1)
        t_m, t_s = round(statistics.mean(r["transfer"]), 2), round(statistics.stdev(r["transfer"]), 2)
        print(f"{name:<16}{f'{s_m} / {s_s}':<22}{f'{t_m} / {t_s}'}")

    print("\n--- Pass/fail check ---")
    ours = results["full (ours)"]
    ql = results["q_learning"]

    ours_beats_ql_survival, (d1, lo1, hi1) = significant_improvement(ours["survival"], ql["survival"])
    ql_beats_ours_survival, (d2, lo2, hi2) = significant_improvement(ql["survival"], ours["survival"])
    ours_beats_ql_transfer, (d3, lo3, hi3) = significant_improvement(ours["transfer"], ql["transfer"])
    ql_beats_ours_transfer, (d4, lo4, hi4) = significant_improvement(ql["transfer"], ours["transfer"])

    print(f"our agent significantly beats Q-learning on survival: {ours_beats_ql_survival} (diff={d1})")
    print(f"Q-learning significantly beats our agent on survival: {ql_beats_ours_survival} (diff={d2})")
    print(f"our agent significantly beats Q-learning on transfer: {ours_beats_ql_transfer} (diff={d3})")
    print(f"Q-learning significantly beats our agent on transfer: {ql_beats_ours_transfer} (diff={d4})")

    print("\nBuilt-in vs. emergent: Q-learning's state space (position x feature) and")
    print("its hyperparameters are hand-chosen, matching the standard applied to this")
    print("project's own untuned choices elsewhere. Q-learning has no built-in")
    print("generalization mechanism across object identity -- transfer to a novel red")
    print("object at an unvisited position requires having visited that exact state")
    print("before, a limitation structurally similar to the memory-only baseline.")

    print("\nNote: the training episode budget may be insufficient for Q-learning's")
    print("larger tabular state space to fully converge -- if it underperforms, that")
    print("may partly reflect a sample-budget mismatch rather than a capability gap.")
    print("Reported at the same episode budget as every other agent in this project,")
    print("not tuned in Q-learning's favor.")


if __name__ == "__main__":
    main()
