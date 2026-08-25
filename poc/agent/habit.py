"""Habit formation: compiling a repeatedly successful action sequence into
a directly executable routine.

The value-iteration planner (agent/value_iteration.py) still only acts on
objects it can currently see -- when nothing relevant is visible, it falls
back to random movement while it waits to stumble across the next step of
its own plan, even after it has already proven, many times, exactly where
that next step is. This module tracks the positions where a successful
sequence's steps occurred and, once a repetition threshold is crossed,
compiles a fixed navigation routine that goes directly to those remembered
positions without waiting for them to come into view.

The compilation trigger (a hit-count threshold) and the compiled routine's
structure (navigate to remembered position, act, repeat) are hand-designed.
What must be learned: WHICH positions get remembered and WHEN the
threshold is crossed -- nothing hand-codes where the key or the reward
object are; both are recorded only after the agent has already reached and
used them successfully on its own.
"""

from collections import defaultdict

from agent.value_iteration import ConceptGraph
from agent.planner import choose_action, step_toward

COMPILE_THRESHOLD = 5  # successful repetitions of a step before it is compiled


class HabitAgent:
    def __init__(self, replan_every=10):
        self.graph = ConceptGraph()
        self.replan_every = replan_every
        self._episodes_since_replan = 0

        # (feature, key_state) -> {position: success_count}
        self._position_success = defaultdict(lambda: defaultdict(int))
        # (feature, key_state) -> compiled target position, once threshold crossed
        self._compiled_targets = {}

    def act(self, obs):
        pos = obs["pos"]
        key_state = obs["has_key"]
        feature = obs["feature"]
        visible = obs["visible"]

        if feature is not None:
            return "EAT"

        compiled_target = self._compiled_targets.get((self._goal_feature(key_state), key_state))
        if compiled_target is not None and compiled_target != pos:
            return step_toward(pos, compiled_target)

        value_fn = lambda f: self.graph.q_value(f, key_state)
        return choose_action(pos, feature, visible, value_fn, threshold=0.05)

    def _goal_feature(self, key_state):
        """Which feature the agent is currently trying to reach, based on
        its own learned values -- not hand-coded to a specific color.
        Picks the feature with the highest learned value for the current
        key state, among features it has actually observed.
        """
        best_feature, best_val = None, float("-inf")
        for feature in self.graph.known_features():
            val = self.graph.q_value(feature, key_state)
            if val > best_val:
                best_feature, best_val = feature, val
        return best_feature

    def observe_result(self, obs, action, next_obs, delta_energy):
        pos = obs["pos"]
        key_state = obs["has_key"]
        feature = obs["feature"]

        if action == "EAT" and feature is not None:
            self.graph.observe(feature, key_state, action, delta_energy, next_obs["has_key"])

            # A step is worth compiling if the agent's own learned value for
            # it is positive -- not whether its IMMEDIATE reward was positive.
            # Using raw delta_energy here would mean an instrumental action
            # with zero direct reward (such as acquiring an enabling object)
            # could never be recognized as worth repeating, even though its
            # propagated value correctly identifies it as valuable.
            if self.graph.q_value(feature, key_state) > 0:
                key = (feature, key_state)
                self._position_success[key][pos] += 1
                if self._position_success[key][pos] >= COMPILE_THRESHOLD:
                    self._compiled_targets[key] = pos

    def end_of_episode(self):
        self._episodes_since_replan += 1
        if self._episodes_since_replan >= self.replan_every:
            self.graph.run_value_iteration()
            self._episodes_since_replan = 0
