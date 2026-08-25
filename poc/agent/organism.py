"""Organism: a single, reusable agent combining every mechanism proven
across this project's experiments, instead of a separate one-off agent
class per experiment.

Prior experiments each hand-wrote their own agent class combining a subset
of: episodic memory, feature-keyed concept learning, context-aware causal
reasoning, value iteration for instrumental value, habit compilation, and
curiosity-driven exploration. That worked for proving each mechanism in
isolation, but meant teaching the system something new required writing a
new agent class every time. Organism composes all of these into one
class, configurable rather than rewritten, so a new task only requires a
new environment -- not new agent code.

Generalizations made here that did not exist in the original per-
experiment modules, needed to make this genuinely reusable:

  - Context is no longer a hardcoded boolean (has_key). It is any
    hashable value the caller's `context_fn` extracts from an
    observation -- the value-iteration MDP discovers its own state set
    from whatever distinct context values it actually observes, rather
    than assuming exactly two states named True/False.
  - Perception is automatic: if a feature is a raw numeric vector
    (tuple/list of numbers), it is passed through an online clusterer to
    obtain a discrete label before anything else touches it. If a feature
    is already a discrete value (a string, an int), it is used directly,
    with no clustering step. Either way the rest of the architecture sees
    only discrete feature labels.
  - Habit compilation and curiosity are both optional, toggled by
    constructor flags, so a task that doesn't need them (most won't) pays
    no extra complexity.

Every individual mechanism here is unchanged in substance from its
original, validated module (agent/concepts.py, agent/value_iteration.py,
agent/habit.py, agent/perception.py, agent/planner.py) -- this class
generalizes and composes them, it does not reimplement them from scratch.
"""

from collections import defaultdict, deque

from agent.memory import EpisodicMemory
from agent.perception import OnlineClusterer
from agent.planner import choose_action, choose_action_curious, choose_action_spatial_curious

GAMMA = 0.9
VI_SWEEPS = 50
WINDOW_SIZE = 40
COMPILE_THRESHOLD = 5


def _is_raw_vector(value):
    return isinstance(value, (tuple, list)) and len(value) > 0 and isinstance(value[0], float)


class _ContextGraph:
    """Generalized version of agent/value_iteration.py's ConceptGraph: an
    arbitrary discrete context set, discovered from observation, instead
    of a hardcoded {False, True}.

    Two distinct kinds of value are computed, deliberately kept separate
    rather than folded into one recursive formula:

      - Direct value: the immediate expected outcome of a (feature,
        context) pair, from its own observed reward history alone. This
        is what every feature gets by default.
      - Instrumental value: propagated value from a DIFFERENT future
        context, added ONLY for a (feature, context) pair with real
        observed evidence that acting on it changes the context to
        something else. A feature whose action leaves the context
        unchanged (a self-loop) never receives propagated value -- its
        Q-value is exactly its direct value, nothing more.

    This distinction is what the earlier, single-formula version of this
    graph lacked: every feature's Q-value recursed through the same
    context's own value estimate regardless of whether that feature ever
    actually caused a transition, letting an inert feature (e.g. a color
    with no effect on anything) inherit inflated value from whichever
    feature happened to have the best value anywhere -- a self-referential
    fixed point (V = R + gamma*V) with no real causal justification. Value
    now only flows backward along an edge that has been observed to exist.
    """

    def __init__(self, window_size=WINDOW_SIZE):
        self.reward_windows = defaultdict(lambda: deque(maxlen=window_size))
        # (feature, context_before) -> {context_after: count}
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.values = {}

    def observe(self, feature, context_before, delta_energy, context_after):
        self.reward_windows[(feature, context_before)].append(delta_energy)
        self.transition_counts[(feature, context_before)][context_after] += 1

    def _mean_reward(self, feature, context):
        w = self.reward_windows.get((feature, context))
        return sum(w) / len(w) if w else 0.0

    def _leads_elsewhere(self, feature, context):
        """Returns {context_after: probability} restricted to context_after
        values DIFFERENT from `context` -- i.e. only the portion of this
        feature's observed transitions that constitute real evidence of
        changing something. Empty if every observed transition was a
        self-loop (or if this (feature, context) has never been observed).
        """
        counts = self.transition_counts.get((feature, context))
        if not counts:
            return {}
        total = sum(counts.values())
        return {c: n / total for c, n in counts.items() if c != context}

    def known_features(self):
        return {f for (f, _) in self.reward_windows.keys()}

    def known_contexts(self):
        contexts = {c for (_, c) in self.reward_windows.keys()}
        for counts in self.transition_counts.values():
            contexts.update(counts.keys())
        return contexts or {None}

    def total_count(self, feature):
        return sum(len(w) for (f, _), w in self.reward_windows.items() if f == feature)

    def _q_value_using(self, feature, context, value_table):
        direct = self._mean_reward(feature, context)
        elsewhere = self._leads_elsewhere(feature, context)
        if not elsewhere:
            return direct
        propagated = sum(p * value_table.get(c, 0.0) for c, p in elsewhere.items())
        return direct + GAMMA * propagated

    def run_value_iteration(self, sweeps=VI_SWEEPS, gamma=GAMMA):
        contexts = self.known_contexts()
        features = self.known_features()
        V = {c: 0.0 for c in contexts}
        for _ in range(sweeps):
            new_V = {}
            for context in contexts:
                best_q = 0.0
                for feature in features:
                    q = self._q_value_using(feature, context, V)
                    if q > best_q:
                        best_q = q
                new_V[context] = best_q
            V = new_V
        self.values = V
        return V

    def q_value(self, feature, context):
        return self._q_value_using(feature, context, self.values)


class Organism:
    """A single reusable agent. Constructor flags turn optional mechanisms
    on or off; context_fn (optional) tells it how to read task-specific
    state (like has_key) out of an observation, generalizing what used to
    require a bespoke agent class per task.
    """

    def __init__(
        self,
        context_fn=None,
        use_habits=True,
        exploration="spatial",  # "random", "curious", or "spatial"
        clustering_threshold=0.5,
        replan_every=10,
    ):
        self.memory = EpisodicMemory()
        self.graph = _ContextGraph()
        self.clusterer = OnlineClusterer(distance_threshold=clustering_threshold)
        self.context_fn = context_fn or (lambda obs: None)
        self.use_habits = use_habits
        self.exploration = exploration
        self.replan_every = replan_every

        self._episodes_since_replan = 0
        self._position_success = defaultdict(lambda: defaultdict(int))
        self._compiled_targets = {}
        self._last_feature_label = None
        self._last_context = None

    def _label(self, raw_feature):
        if raw_feature is None:
            return None
        if _is_raw_vector(raw_feature):
            return self.clusterer.classify(raw_feature)
        return raw_feature

    def _goal_feature(self, context):
        best_feature, best_val = None, float("-inf")
        for feature in self.graph.known_features():
            val = self.graph.q_value(feature, context)
            if val > best_val:
                best_feature, best_val = feature, val
        return best_feature

    def act(self, obs):
        pos = obs["pos"]
        context = self.context_fn(obs)
        feature = self._label(obs["feature"])
        self._last_feature_label = feature
        self._last_context = context
        visible = {p: self._label(v) for p, v in obs["visible"].items()}

        if feature is not None:
            return "EAT"

        if self.use_habits:
            compiled_target = self._compiled_targets.get((self._goal_feature(context), context))
            if compiled_target is not None and compiled_target != pos:
                from agent.planner import step_toward
                return step_toward(pos, compiled_target)

        value_fn = lambda f: self.graph.q_value(f, context)

        if self.exploration == "curious":
            count_fn = lambda f: self.graph.total_count(f)
            return choose_action_curious(pos, feature, visible, value_fn, count_fn, threshold=0.05)
        if self.exploration == "spatial":
            object_count_fn = lambda f: self.graph.total_count(f)
            cell_count_fn = lambda p: self._cell_visits.get(p, 0) if hasattr(self, "_cell_visits") else 0
            return choose_action_spatial_curious(
                pos, feature, visible, value_fn, object_count_fn, cell_count_fn, threshold=0.05
            )
        return choose_action(pos, feature, visible, value_fn, threshold=0.05)

    def observe_result(self, obs, action, next_obs, delta_energy):
        self.memory.record(obs, action, next_obs, delta_energy)
        if not hasattr(self, "_cell_visits"):
            self._cell_visits = defaultdict(int)
        self._cell_visits[obs["pos"]] += 1

        if action != "EAT" or self._last_feature_label is None:
            return

        next_context = self.context_fn(next_obs)
        self.graph.observe(self._last_feature_label, self._last_context, delta_energy, next_context)

        if self.use_habits and self.graph.q_value(self._last_feature_label, self._last_context) > 0:
            key = (self._last_feature_label, self._last_context)
            pos = obs["pos"]
            self._position_success[key][pos] += 1
            if self._position_success[key][pos] >= COMPILE_THRESHOLD:
                self._compiled_targets[key] = pos

    def end_of_episode(self):
        self._episodes_since_replan += 1
        if self._episodes_since_replan >= self.replan_every:
            self.graph.run_value_iteration()
            self._episodes_since_replan = 0
