"""Ablation agents for the gridworld task: a random baseline, a
memory-only baseline with no generalization, and the full concept-table
agent. Comparing FullAgent against MemoryOnlyAgent isolates a single
variable: whether grouping outcomes by feature, instead of by exact
observed instance, allows the agent to act correctly on an object it has
never previously encountered.
"""

import random

from agent.memory import EpisodicMemory
from agent.concepts import ConceptTable
from agent.planner import choose_action
from env.gridworld import ACTIONS, MOVES


class RandomAgent:
    """No memory, no learning. Acts randomly, eats whatever it's standing on."""

    def __init__(self):
        self.memory = EpisodicMemory()

    def act(self, obs):
        if obs["feature"] is not None:
            return "EAT"
        return random.choice(list(MOVES.keys()))

    def observe_result(self, obs, action, next_obs, delta_energy):
        self.memory.record(obs, action, next_obs, delta_energy)


class MemoryOnlyAgent:
    """Remembers exact (position -> outcome) from past EATs. No feature-level
    aggregation: seeing a same-colored object at a NEW position gives it no
    information, since it only ever indexes by exact position, not by color.
    This is the control that isolates what feature-grouping (FullAgent) buys.
    """

    def __init__(self):
        self.memory = EpisodicMemory()
        self.known_good_positions = {}  # exact pos -> expected delta_energy

    def act(self, obs):
        pos = obs["pos"]
        if obs["feature"] is not None:
            return "EAT"

        best_target, best_val = None, 1.0
        for opos, val in self.known_good_positions.items():
            if val > best_val:
                best_target, best_val = opos, val
        if best_target is not None and best_target != pos:
            from agent.planner import step_toward
            return step_toward(pos, best_target)

        return random.choice(list(MOVES.keys()))

    def observe_result(self, obs, action, next_obs, delta_energy):
        self.memory.record(obs, action, next_obs, delta_energy)
        if action == "EAT":
            self.known_good_positions[obs["pos"]] = delta_energy


class FullAgent:
    """Groups outcomes by feature (color), not by object identity. This is
    the abstraction step: once red+EAT is known good, ANY red object it can
    currently SEE (via vision, not memory of that exact spot) gets sought
    out and eaten -- including one it has never encountered before.
    """

    def __init__(self):
        self.memory = EpisodicMemory()
        self.concepts = ConceptTable()

    def act(self, obs):
        value_fn = lambda feature: self.concepts.expected_value(feature, "EAT")
        return choose_action(obs["pos"], obs["feature"], obs["visible"], value_fn)

    def observe_result(self, obs, action, next_obs, delta_energy):
        self.memory.record(obs, action, next_obs, delta_energy)
        if action == "EAT" and obs["feature"] is not None:
            self.concepts.update(obs["feature"], action, delta_energy)
