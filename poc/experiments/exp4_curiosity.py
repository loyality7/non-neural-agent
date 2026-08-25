"""Exploration-strategy comparison: random, object-novelty, and spatial
novelty.

Compares three exploration policies on a hard variant of the conjunctive
environment where the instrumentally valuable object (the key) is placed
outside the agent's vision range at spawn, so locating it requires
directed search rather than incidental discovery:

  random    -- uniform random movement when nothing visible is worth
               acting on.
  curious   -- prefers the least-sampled currently visible object.
  spatial   -- additionally prefers the least-visited neighboring cell
               when nothing is currently visible at all.

The comparison measures sample efficiency -- competence reached after a
given number of training episodes -- rather than only final performance,
so all three agents are evaluated at multiple checkpoints during training.
"""

import random
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict

from env.conjunctive_world import fresh_conjunctive_world_hard, RED, YELLOW
from agent.value_iteration import ConceptGraph
from agent.planner import choose_action, choose_action_curious, choose_action_spatial_curious
from stats_utils import significant_improvement, cohens_d

from experiments.exp2_conjunctive_rule import run_episode, correct_behavior_test

N_SEEDS = 20
CHECKPOINTS = [30, 60, 100, 150, 200]  # episodes -- measure sample efficiency, not just endpoint
VI_REPLAN_EVERY = 10
MODES = ("random", "curious", "spatial")


class ValueIterationAgentBase:
    """Shared agent machinery across all three exploration modes -- an
    identical concept graph and value-iteration mechanism, differing only
    in which planner function act() calls. This isolates the exploration
    policy as the sole variable under comparison.

    mode: one of "random", "curious", or "spatial" (see module docstring).
    """

    def __init__(self, mode):
        assert mode in MODES
        self.graph = ConceptGraph()
        self.mode = mode
        self._episodes_since_replan = 0
        self.cell_visits = defaultdict(int)

    def act(self, obs):
        pos = obs["pos"]
        key_state = obs["has_key"]
        feature = obs["feature"]
        visible = obs["visible"]
        value_fn = lambda f: self.graph.q_value(f, key_state)
        if self.mode == "curious":
            count_fn = lambda f: self.graph.total_count(f)
            return choose_action_curious(pos, feature, visible, value_fn, count_fn, threshold=0.05)
        if self.mode == "spatial":
            object_count_fn = lambda f: self.graph.total_count(f)
            cell_count_fn = lambda p: self.cell_visits.get(p, 0)
            return choose_action_spatial_curious(
                pos, feature, visible, value_fn, object_count_fn, cell_count_fn, threshold=0.05
            )
        return choose_action(pos, feature, visible, value_fn, threshold=0.05)

    def observe_result(self, obs, action, next_obs, delta_energy):
        self.cell_visits[obs["pos"]] += 1
        if action == "EAT" and obs["feature"] is not None:
            self.graph.observe(
                obs["feature"], obs["has_key"], action, delta_energy, next_obs["has_key"]
            )

    def end_of_episode(self):
        self._episodes_since_replan += 1
        if self._episodes_since_replan >= VI_REPLAN_EVERY:
            self.graph.run_value_iteration()
            self._episodes_since_replan = 0


def correct_behavior_test_hard(agent, trials=20, step_budget=150):
    """Same success criterion as correct_behavior_test (eat red only after
    the key), evaluated on the far-key layout -- tests whether the agent
    can locate a key outside its initial vision range, not only whether it
    acts correctly once already standing next to one.
    """
    successes = 0
    for _ in range(trials):
        world = fresh_conjunctive_world_hard()
        world.objects.clear()
        world.add_object((0, 0), YELLOW)
        world.add_object((7, 7), RED)
        world.agent_pos = (4, 4)
        world.energy = 80
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


def train_with_checkpoints(mode, checkpoints=CHECKPOINTS):
    agent = ValueIterationAgentBase(mode=mode)
    results_at = {}
    max_ep = max(checkpoints)
    for ep in range(1, max_ep + 1):
        world = fresh_conjunctive_world_hard()
        run_episode(agent, world)
        agent.end_of_episode()
        if ep in checkpoints:
            agent.graph.run_value_iteration()
            success = correct_behavior_test_hard(agent, trials=20)
            results_at[ep] = success
    return results_at


def run_one_seed(seed):
    results = {}
    for mode in MODES:
        random.seed(seed)  # identical environment sequence across modes
        results[mode] = train_with_checkpoints(mode=mode)
    return results


def main():
    all_results = {mode: {ep: [] for ep in CHECKPOINTS} for mode in MODES}

    for seed in range(N_SEEDS):
        seed_results = run_one_seed(seed)
        for mode in MODES:
            for ep in CHECKPOINTS:
                all_results[mode][ep].append(seed_results[mode][ep])

    print(f"\n=== Exploration Strategy Comparison ({N_SEEDS} seeds) ===\n")
    print("random   = uniform random movement, no exploration bias")
    print("curious  = prefers the least-sampled visible object")
    print("spatial  = also prefers the least-visited neighboring cell\n")
    print("correct-behavior rate (eat red only after the key) by training episodes so far:")
    header = f"{'episodes':<10}" + "".join(f"{m + ' (mean/std)':<20}" for m in MODES)
    print(header)
    print("-" * len(header))

    for ep in CHECKPOINTS:
        row = f"{ep:<10}"
        for mode in MODES:
            vals = all_results[mode][ep]
            row += f"{f'{round(statistics.mean(vals), 2)} / {round(statistics.stdev(vals), 2)}':<20}"
        print(row)

    print("\n--- Pass/fail check ---")
    final_ep = CHECKPOINTS[-1]
    spatial_final = all_results["spatial"][final_ep]
    curious_final = all_results["curious"][final_ep]
    random_final = all_results["random"][final_ep]

    spatial_beats_curious, (d1, lo1, hi1) = significant_improvement(spatial_final, curious_final)
    spatial_beats_random, (d2, lo2, hi2) = significant_improvement(spatial_final, random_final)
    d_curious = cohens_d(spatial_final, curious_final)
    d_random = cohens_d(spatial_final, random_final)

    print(f"spatial beats curious at final checkpoint ({final_ep} eps): {spatial_beats_curious} "
          f"(diff={d1}, 95% CI=[{lo1},{hi1}], Cohen's d={d_curious})")
    print(f"spatial beats random at final checkpoint ({final_ep} eps): {spatial_beats_random} "
          f"(diff={d2}, 95% CI=[{lo2},{hi2}], Cohen's d={d_random})")

    print("\nBuilt-in vs. emergent: the visited-cell counting rule and the")
    print("'prefer the least-visited neighbor' fallback policy are hand-designed.")
    print("What's emergent: the actual visit counts (which cells get explored")
    print("when), and therefore which specific direction gets prioritized moment to moment.")

    if spatial_beats_curious and spatial_beats_random:
        print("\nRESULT: PASS -- spatial novelty closes the gap object-only curiosity")
        print("could not: the agent searches unexplored regions, not only currently")
        print("visible objects.")
    else:
        print("\nRESULT: FAIL -- the spatial exploration strategy did not produce a")
        print("measurable advantage over object-only curiosity and/or random movement.")


if __name__ == "__main__":
    main()
