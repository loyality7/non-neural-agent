"""Raw-perception experiment.

Every prior experiment gave the agent a clean color string as its
observation. This tests the same entity-discovery-and-transfer task as
the first experiment, but with the color string replaced by a noisy
multi-attribute numeric vector -- the agent must derive its own stable
categories from noise before any causal learning can even begin.

Two agents on the noisy-perception environment:
  oracle      -- FullAgent, given the true color string directly (as in
                 every previous experiment). Establishes the ceiling: what
                 performance looks like when perception is not a problem.
  perception  -- PerceptionAgent, given only noisy numeric vectors and an
                 online clusterer it must use to discover its own
                 categories. Nothing about which vectors belong together
                 is given.

If perception matches oracle's performance, self-discovered categories
under noise are sufficient. If it falls short, that is the expected,
literature-predicted result -- and precisely where a learned (likely
neural) perceptual front-end becomes the honest next step, exactly as
flagged from the start of this project.
"""

import random
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.perception_world import fresh_perception_world, TRUE_ATTRIBUTES
from env.gridworld import RED
from agent.agent import FullAgent
from agent.perception_agent import PerceptionAgent

TRAIN_EPISODES = 200
N_SEEDS = 20
NOVEL_RED_POS = (7, 0)


class OracleAgent(FullAgent):
    """FullAgent expects a color string as obs["feature"]; the noisy world
    provides a numeric vector instead. This wraps observations to strip
    away the noise and hand FullAgent the ground-truth color, exactly as
    every previous experiment did -- the ceiling condition, not a new agent.
    """

    def __init__(self):
        super().__init__()
        self._reverse_lookup = {v: k for k, v in TRUE_ATTRIBUTES.items()}

    def _closest_true_color(self, vector):
        best_color, best_dist = None, float("inf")
        for color, true_vec in TRUE_ATTRIBUTES.items():
            d = sum((a - b) ** 2 for a, b in zip(vector, true_vec)) ** 0.5
            if d < best_dist:
                best_color, best_dist = color, d
        return best_color

    def _decode(self, obs):
        feature = self._closest_true_color(obs["feature"]) if obs["feature"] is not None else None
        visible = {p: self._closest_true_color(v) for p, v in obs["visible"].items()}
        return {"pos": obs["pos"], "feature": feature, "visible": visible}

    def act(self, obs):
        return super().act(self._decode(obs))

    def observe_result(self, obs, action, next_obs, delta_energy):
        super().observe_result(self._decode(obs), action, self._decode(next_obs), delta_energy)


def run_episode(agent, world):
    obs = world.observe()
    while world.alive():
        action = agent.act(obs)
        next_obs, delta_energy, done = world.step(action)
        agent.observe_result(obs, action, next_obs, delta_energy)
        obs = next_obs
        if done:
            break
    return world.steps


def train(agent_cls, episodes=TRAIN_EPISODES):
    agent = agent_cls()
    survival = []
    for _ in range(episodes):
        world = fresh_perception_world()
        steps = run_episode(agent, world)
        survival.append(steps)
    return agent, survival


def transfer_test(agent, trials=30, step_budget=80):
    successes = 0
    for _ in range(trials):
        world = fresh_perception_world()
        world.objects.clear()
        world.add_object(NOVEL_RED_POS, RED)
        world.agent_pos = (0, 0)
        world.energy = 50
        world.steps = 0

        obs = world.observe()
        for _ in range(step_budget):
            was_on_object = obs["feature"] is not None
            action = agent.act(obs)
            next_obs, delta_energy, done = world.step(action)
            if action == "EAT" and was_on_object and delta_energy > 0:
                successes += 1
                break
            obs = next_obs
            if done:
                break
    return successes / trials


def run_one_seed(seed):
    random.seed(seed)
    oracle_agent, oracle_survival = train(OracleAgent)
    random.seed(seed)
    perception_agent, perception_survival = train(PerceptionAgent)

    oracle_transfer = transfer_test(oracle_agent)
    perception_transfer = transfer_test(perception_agent)

    return {
        "oracle_survival": sum(oracle_survival) / len(oracle_survival),
        "perception_survival": sum(perception_survival) / len(perception_survival),
        "oracle_transfer": oracle_transfer,
        "perception_transfer": perception_transfer,
        "num_clusters_discovered": perception_agent.clusterer.num_clusters(),
    }


def main():
    results = [run_one_seed(s) for s in range(N_SEEDS)]

    oracle_surv = [r["oracle_survival"] for r in results]
    perception_surv = [r["perception_survival"] for r in results]
    oracle_trans = [r["oracle_transfer"] for r in results]
    perception_trans = [r["perception_transfer"] for r in results]
    num_clusters = [r["num_clusters_discovered"] for r in results]

    print(f"\n=== Raw Perception: Self-Discovered Categories from Noise ({N_SEEDS} seeds) ===\n")
    print(f"{'agent':<14}{'survival (mean/std)':<22}{'transfer_rate (mean/std)'}")
    print("-" * 60)
    print(f"{'oracle':<14}{f'{round(statistics.mean(oracle_surv), 1)} / {round(statistics.stdev(oracle_surv), 1)}':<22}"
          f"{round(statistics.mean(oracle_trans), 2)} / {round(statistics.stdev(oracle_trans), 2)}")
    print(f"{'perception':<14}{f'{round(statistics.mean(perception_surv), 1)} / {round(statistics.stdev(perception_surv), 1)}':<22}"
          f"{round(statistics.mean(perception_trans), 2)} / {round(statistics.stdev(perception_trans), 2)}")

    print(f"\nnumber of clusters discovered (true number of colors is 3), mean/std: "
          f"{round(statistics.mean(num_clusters), 2)} / {round(statistics.stdev(num_clusters), 2)}")

    from stats_utils import significant_improvement, cohens_d
    oracle_beats_perception, (diff, lo, hi) = significant_improvement(oracle_trans, perception_trans)
    d = cohens_d(oracle_trans, perception_trans)

    print("\n--- Pass/fail check ---")
    print(f"oracle significantly beats perception on transfer: {oracle_beats_perception} "
          f"(diff={diff}, 95% CI=[{lo},{hi}], Cohen's d={d})")
    correct_cluster_count = abs(statistics.mean(num_clusters) - 3) < 0.5

    print("\nBuilt-in vs. emergent: the true attribute vectors per color and the noise")
    print("level are environment ground truth, never exposed to the agent. The distance")
    print("threshold for the clustering step is hand-chosen. Emergent: the actual")
    print("clusters discovered, how many there are, and whether they align with the")
    print("true colors well enough to support the same causal learning as before.")

    if not oracle_beats_perception and correct_cluster_count:
        print("\nRESULT: PASS -- self-discovered categories from noisy signals support")
        print("the same concept formation and transfer as ground-truth categories.")
    elif correct_cluster_count:
        print("\nRESULT: PARTIAL -- clustering correctly discovers the true category count,")
        print("but downstream performance still falls measurably short of the oracle.")
    else:
        print("\nRESULT: FAIL -- clustering did not reliably discover the true category")
        print("structure. Non-neural perception is a historically likely point for this")
        print("class of architecture to break; report exactly this, not a disguised success.")


if __name__ == "__main__":
    main()
