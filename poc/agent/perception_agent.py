"""Agent that must discover its own object categories from noisy numeric
observations before it can apply the same causal-learning machinery used
throughout this project.

Every observed vector (current cell and every visible object) is first
passed through an OnlineClusterer to obtain a cluster label. From that
point on, the architecture is identical to FullAgent: a ConceptTable keyed
by (cluster label, action), and the same greedy planner. The only new
variable under test is whether self-discovered cluster labels, instead of
a hand-given color string, still support correct concept formation and
transfer.
"""

from agent.memory import EpisodicMemory
from agent.concepts import ConceptTable
from agent.perception import OnlineClusterer
from agent.planner import choose_action


class PerceptionAgent:
    def __init__(self, distance_threshold=0.5):
        self.memory = EpisodicMemory()
        self.concepts = ConceptTable()
        self.clusterer = OnlineClusterer(distance_threshold=distance_threshold)
        self._last_feature_label = None

    def act(self, obs):
        pos = obs["pos"]
        feature = self.clusterer.classify(obs["feature"]) if obs["feature"] is not None else None
        self._last_feature_label = feature
        visible = {p: self.clusterer.classify(v) for p, v in obs["visible"].items()}
        value_fn = lambda f: self.concepts.expected_value(f, "EAT")
        return choose_action(pos, feature, visible, value_fn)

    def observe_result(self, obs, action, next_obs, delta_energy):
        self.memory.record(obs, action, next_obs, delta_energy)
        if action == "EAT" and self._last_feature_label is not None:
            self.concepts.update(self._last_feature_label, action, delta_energy)
