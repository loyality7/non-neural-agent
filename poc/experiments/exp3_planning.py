"""Instrumental-value planning experiment.

Even with a correctly conflict-free conjunctive concept representation
(red is valuable only while holding the key), a purely greedy planner will
never deliberately seek the key itself, since it pays zero direct reward
and never clears the planner's value threshold on its own. This experiment
compares three agents on the conjunctive-rule environment:

  flat          -- the unmodified FullAgent, blind to key context entirely.
  conjunctive   -- the correct-representation agent from the conjunctive
                   rule experiment, still using greedy action selection.
                   Demonstrates the instrumental-value gap directly.
  value_iter    -- the same conjunctive feature representation, with value
                   iteration over a learned two-state MDP so the key's
                   value includes the future reward it enables, despite
                   zero immediate reward of its own.

The reward table and the key's effect on state are both estimated purely
from the value_iter agent's own observations; no domain-specific knowledge
of the key's role is provided.
"""

import random
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.conjunctive_world import fresh_conjunctive_world, RED, YELLOW
from agent.agent import FullAgent
from agent.value_iteration import ConceptGraph
from agent.planner import choose_action
from stats_utils import significant_improvement, cohens_d

from experiments.exp2_conjunctive_rule import ConjunctiveAgent, run_episode, correct_behavior_test

TRAIN_EPISODES = 300
N_SEEDS = 20
VI_REPLAN_EVERY = 10  # episodes between value-iteration sweeps


class ValueIterationAgent:
    def __init__(self):
        self.graph = ConceptGraph()
        self._episodes_since_replan = 0

    def act(self, obs):
        pos = obs["pos"]
        key_state = obs["has_key"]
        feature = obs["feature"]
        visible = obs["visible"]
        value_fn = lambda f: self.graph.q_value(f, key_state)
        return choose_action(pos, feature, visible, value_fn, threshold=0.05)

    def observe_result(self, obs, action, next_obs, delta_energy):
        if action == "EAT" and obs["feature"] is not None:
            self.graph.observe(
                obs["feature"], obs["has_key"], action, delta_energy, next_obs["has_key"]
            )

    def end_of_episode(self):
        self._episodes_since_replan += 1
        if self._episodes_since_replan >= VI_REPLAN_EVERY:
            self.graph.run_value_iteration()
            self._episodes_since_replan = 0


def train_vi(episodes=TRAIN_EPISODES):
    agent = ValueIterationAgent()
    survival = []
    for _ in range(episodes):
        world = fresh_conjunctive_world()
        steps = run_episode(agent, world)
        survival.append(steps)
        agent.end_of_episode()
    agent.graph.run_value_iteration()  # final sweep prior to evaluation
    return agent, survival


def train_generic(agent_cls, episodes=TRAIN_EPISODES):
    agent = agent_cls()
    survival = []
    for _ in range(episodes):
        world = fresh_conjunctive_world()
        steps = run_episode(agent, world)
        survival.append(steps)
    return agent, survival


def run_one_seed(seed):
    random.seed(seed)
    flat_agent, _ = train_generic(FullAgent)
    random.seed(seed)
    conj_agent, _ = train_generic(ConjunctiveAgent)
    random.seed(seed)
    vi_agent, _ = train_vi()

    flat_success = correct_behavior_test(flat_agent, is_conjunctive=False)
    conj_success = correct_behavior_test(conj_agent, is_conjunctive=True)
    vi_success = correct_behavior_test(vi_agent, is_conjunctive=True)

    return {
        "flat_success": flat_success,
        "conj_success": conj_success,
        "vi_success": vi_success,
        "vi_values": dict(vi_agent.graph.values),
        "vi_q_yellow_no_key": vi_agent.graph.q_value(YELLOW, False),
        "vi_q_red_no_key": vi_agent.graph.q_value(RED, False),
    }


def main():
    results = [run_one_seed(s) for s in range(N_SEEDS)]

    flat_succ = [r["flat_success"] for r in results]
    conj_succ = [r["conj_success"] for r in results]
    vi_succ = [r["vi_success"] for r in results]

    print(f"\n=== Instrumental-Value Planning (Value Iteration) ({N_SEEDS} seeds) ===\n")
    print("correct-behavior rate (eat red only after grabbing the key):")
    print(f"  flat (no key concept at all):        "
          f"{round(statistics.mean(flat_succ), 2)} / {round(statistics.stdev(flat_succ), 2)}")
    print(f"  conjunctive (correct rep, greedy):    "
          f"{round(statistics.mean(conj_succ), 2)} / {round(statistics.stdev(conj_succ), 2)}")
    print(f"  value_iter (correct rep, VI planning):"
          f" {round(statistics.mean(vi_succ), 2)} / {round(statistics.stdev(vi_succ), 2)}")

    print(f"\nvalue_iter's learned Q(eat yellow | no key) -- expected to be positive")
    print(f"despite yellow's own direct reward being zero, if value propagation works:")
    q_yellow = [r["vi_q_yellow_no_key"] for r in results]
    q_red = [r["vi_q_red_no_key"] for r in results]
    print(f"  Q(yellow, key=False) mean/std: {round(statistics.mean(q_yellow), 2)} / {round(statistics.stdev(q_yellow), 2)}")
    print(f"  Q(red,    key=False) mean/std: {round(statistics.mean(q_red), 2)} / {round(statistics.stdev(q_red), 2)}")
    print(f"example V(has_key) (seed 0): {results[0]['vi_values']}")

    print("\n--- Pass/fail check ---")
    vi_beats_conj, (diff, lo, hi) = significant_improvement(vi_succ, conj_succ)
    vi_beats_flat, _ = significant_improvement(vi_succ, flat_succ)
    d = cohens_d(vi_succ, conj_succ)
    baseline_ceiling = statistics.mean(conj_succ)
    above_ceiling = statistics.mean(vi_succ) > baseline_ceiling + 0.10

    print(f"value_iter beats conjunctive-greedy, significant: {vi_beats_conj} "
          f"(diff={diff}, 95% CI=[{lo},{hi}], Cohen's d={d})")
    print(f"value_iter beats flat, significant: {vi_beats_flat}")
    print(f"value_iter mean success rate {round(statistics.mean(vi_succ), 2)} "
          f"clears the greedy-planner ceiling with margin: {above_ceiling}")

    print("\nBuilt-in vs. emergent: the two-state MDP structure and the")
    print("value-iteration algorithm are hand-designed. The reward table and")
    print("the key's effect on state are estimated purely from the agent's")
    print("own observations -- the key's instrumental role is not given directly.")

    if vi_beats_conj and vi_beats_flat and above_ceiling:
        print("\nRESULT: PASS -- value iteration closes the instrumental-value gap")
        print("that greedy planning, even with correct representation, could not.")
    else:
        print("\nRESULT: FAIL -- value iteration did not clear the gate.")


if __name__ == "__main__":
    main()
