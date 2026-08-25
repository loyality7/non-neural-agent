"""Conjunctive (two-condition) rule experiment.

Combinatorial explosion in representing multi-condition rules is a
well-documented failure mode for non-gradient concept-learning systems.
This experiment tests whether a single-feature concept table can still
approximate a rule requiring two conditions, and whether extending the
feature representation resolves it. Two agents are compared:

  flat          -- the unmodified FullAgent. Its feature is color alone,
                   blind to whether the agent is holding the key. Expected
                   to conflate "red with key" and "red without key" into a
                   single diluted average per color.
  conjunctive   -- identical concept-table and planner machinery, but the
                   feature string is extended to f"{color}|key={has_key}",
                   giving the two cases independent estimates.
"""

import random
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.conjunctive_world import fresh_conjunctive_world, RED, YELLOW
from agent.agent import FullAgent, RandomAgent
from agent.concepts import ConceptTable
from agent.planner import choose_action
from stats_utils import significant_improvement, cohens_d

TRAIN_EPISODES = 300
N_SEEDS = 20
NOVEL_RED_WITH_KEY_TRIALS = 30


class ConjunctiveAgent:
    """Agent using the same ConceptTable and planner machinery as
    FullAgent, but with its feature representation augmented by the
    agent's current has_key state. This gives "red while holding the key"
    and "red without the key" independent, separately-tracked estimates
    instead of one diluted average, at the cost of still using a purely
    greedy planner -- it correctly identifies the value of the conjunctive
    rule but cannot yet seek out a zero-direct-reward enabling object on
    purpose, which downstream planning experiments address.
    """

    def __init__(self):
        self.concepts = ConceptTable()

    def _augment(self, color, has_key):
        if color is None:
            return None
        return f"{color}|key={has_key}"

    def act(self, obs):
        pos = obs["pos"]
        feature = self._augment(obs["feature"], obs["has_key"])
        visible = {p: self._augment(c, obs["has_key"]) for p, c in obs["visible"].items()}
        value_fn = lambda f: self.concepts.expected_value(f, "EAT")
        return choose_action(pos, feature, visible, value_fn)

    def observe_result(self, obs, action, next_obs, delta_energy):
        if action == "EAT" and obs["feature"] is not None:
            feature = self._augment(obs["feature"], obs["has_key"])
            self.concepts.update(feature, "EAT", delta_energy)


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
        world = fresh_conjunctive_world()
        steps = run_episode(agent, world)
        survival.append(steps)
    return agent, survival


def correct_behavior_test(agent, is_conjunctive, trials=NOVEL_RED_WITH_KEY_TRIALS, step_budget=100):
    """Tests whether the agent, having already grabbed the key, correctly
    and reliably proceeds to eat a novel red object. The key is placed one
    step from spawn, so acquiring it is not itself a limiting factor.
    """
    successes = 0
    for _ in range(trials):
        world = fresh_conjunctive_world()
        world.objects.clear()
        world.add_object((4, 3), YELLOW)
        world.add_object((7, 0), RED)
        world.agent_pos = (4, 4)
        world.energy = 50
        world.steps = 0
        world.has_key = False

        obs = world.observe()
        for _ in range(step_budget):
            was_on_red_with_key = obs["feature"] == RED and obs["has_key"]
            action = agent.act(obs)
            next_obs, delta_energy, done = world.step(action)
            if action == "EAT" and was_on_red_with_key and delta_energy > 0:
                successes += 1
                break
            obs = next_obs
            if done:
                break
    return successes / trials


def run_one_seed(seed):
    random.seed(seed)
    flat_agent, flat_survival = train(FullAgent)
    random.seed(seed)  # same env sequence for a fair side-by-side
    conj_agent, conj_survival = train(ConjunctiveAgent)
    random.seed(seed)
    rand_agent, rand_survival = train(RandomAgent)

    flat_success = correct_behavior_test(flat_agent, is_conjunctive=False)
    conj_success = correct_behavior_test(conj_agent, is_conjunctive=True)
    rand_success = correct_behavior_test(rand_agent, is_conjunctive=False)

    return {
        "flat_survival": sum(flat_survival) / len(flat_survival),
        "conj_survival": sum(conj_survival) / len(conj_survival),
        "rand_survival": sum(rand_survival) / len(rand_survival),
        "flat_success": flat_success,
        "conj_success": conj_success,
        "rand_success": rand_success,
        "flat_concepts": dict(flat_agent.concepts.stats),
        "conj_concepts": dict(conj_agent.concepts.stats),
    }


def main():
    results = [run_one_seed(s) for s in range(N_SEEDS)]

    flat_succ = [r["flat_success"] for r in results]
    conj_succ = [r["conj_success"] for r in results]
    rand_succ = [r["rand_success"] for r in results]

    print(f"\n=== Conjunctive (Two-Condition) Rule ({N_SEEDS} seeds) ===\n")
    print(f"correct-behavior rate (eat red ONLY after grabbing key):")
    print(f"  flat (color only, blind to key):     "
          f"{round(statistics.mean(flat_succ), 2)} / {round(statistics.stdev(flat_succ), 2)}")
    print(f"  conjunctive (color + key context):   "
          f"{round(statistics.mean(conj_succ), 2)} / {round(statistics.stdev(conj_succ), 2)}")
    print(f"  random baseline:                     "
          f"{round(statistics.mean(rand_succ), 2)} / {round(statistics.stdev(rand_succ), 2)}")

    print(f"\nexample flat concept table (seed 0):")
    print(f"  {results[0]['flat_concepts']}")
    print(f"example conjunctive concept table (seed 0):")
    print(f"  {results[0]['conj_concepts']}")

    print("\n--- Pass/fail check ---")
    conj_beats_flat, (diff, lo, hi) = significant_improvement(conj_succ, flat_succ)
    conj_beats_rand, _ = significant_improvement(conj_succ, rand_succ)
    d = cohens_d(conj_succ, flat_succ)

    print(f"conjunctive beats flat, significant: {conj_beats_flat} "
          f"(diff={diff}, 95% CI=[{lo},{hi}], Cohen's d={d})")
    print(f"conjunctive beats random, significant: {conj_beats_rand}")
    print(f"flat mean success rate: {round(statistics.mean(flat_succ), 2)} "
          f"(a low rate indicates the single-feature representation cannot "
          f"capture the conjunctive rule)")

    if conj_beats_flat and conj_beats_rand:
        print("\nRESULT: PASS -- the extended feature representation successfully")
        print("captures the two-condition rule.")
    else:
        print("\nRESULT: FAIL -- the extended representation did not reliably")
        print("capture the conjunctive rule. See the concept tables above for detail.")


if __name__ == "__main__":
    main()
