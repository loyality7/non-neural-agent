"""Multi-concept interference experiment.

Every prior experiment tested one rule (or one conjunctive rule) at a
time. This tests a different axis: can the agent hold two independent,
unrelated rules simultaneously -- red is rewarding, blue is penalized --
without either estimate being corrupted by the presence of the other?

Three training conditions per seed:
  single_red   -- only the red-is-good rule is active (blue neutral).
  single_blue  -- only the blue-is-bad rule is active (red neutral).
  multi        -- both rules active at once, in the same world.

If the concept table is free of interference, the agent's learned value
for red in the multi condition should match its value for red in the
single_red condition (and likewise for blue), since the two rules concern
entirely different, independently-tracked (feature, action) keys.
"""

import random
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gridworld import fresh_training_world, RED, BLUE, GREEN
from agent.agent import FullAgent
from agent.planner import choose_action
from stats_utils import significant_improvement, cohens_d, bootstrap_diff_ci

TRAIN_EPISODES = 200
N_SEEDS = 20
HEAD_TO_HEAD_TRIALS = 30


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


def train(good_color, bad_color, episodes=TRAIN_EPISODES):
    agent = FullAgent()
    for _ in range(episodes):
        world = fresh_training_world(good_color=good_color, bad_color=bad_color)
        run_episode(agent, world)
    return agent


def head_to_head_red_vs_blue(agent, trials=HEAD_TO_HEAD_TRIALS, step_budget=80):
    """Place a novel red and a novel blue object equidistant from the
    agent. A non-interfering agent should reliably choose red (rewarding)
    over blue (penalized) -- this is a behavioral check, not just a check
    of the raw stored values.
    """
    red_wins, blue_wins = 0, 0
    for _ in range(trials):
        world = fresh_training_world()
        world.objects.clear()
        world.add_object((0, 6), RED)
        world.add_object((6, 0), BLUE)
        world.agent_pos = (3, 3)
        world.energy = 50
        world.steps = 0
        obs = world.observe()
        outcome = None
        for _ in range(step_budget):
            eaten_color = obs["feature"]
            action = agent.act(obs)
            next_obs, delta_energy, done = world.step(action)
            if action == "EAT" and eaten_color is not None:
                outcome = eaten_color
                break
            obs = next_obs
            if done:
                break
        if outcome == RED:
            red_wins += 1
        elif outcome == BLUE:
            blue_wins += 1
    return red_wins / trials, blue_wins / trials


def run_one_seed(seed):
    random.seed(seed)
    single_red_agent = train(good_color=RED, bad_color=None)
    random.seed(seed)
    single_blue_agent = train(good_color=None, bad_color=BLUE)
    random.seed(seed)
    multi_agent = train(good_color=RED, bad_color=BLUE)

    red_single = single_red_agent.concepts.expected_value(RED, "EAT")
    blue_single = single_blue_agent.concepts.expected_value(BLUE, "EAT")
    red_multi = multi_agent.concepts.expected_value(RED, "EAT")
    blue_multi = multi_agent.concepts.expected_value(BLUE, "EAT")
    green_multi = multi_agent.concepts.expected_value(GREEN, "EAT")

    red_h2h, blue_h2h = head_to_head_red_vs_blue(multi_agent)

    return {
        "red_single": red_single,
        "blue_single": blue_single,
        "red_multi": red_multi,
        "blue_multi": blue_multi,
        "green_multi": green_multi,
        "red_h2h": red_h2h,
        "blue_h2h": blue_h2h,
    }


def main():
    results = [run_one_seed(s) for s in range(N_SEEDS)]

    red_single = [r["red_single"] for r in results]
    blue_single = [r["blue_single"] for r in results]
    red_multi = [r["red_multi"] for r in results]
    blue_multi = [r["blue_multi"] for r in results]
    green_multi = [r["green_multi"] for r in results]
    red_h2h = [r["red_h2h"] for r in results]
    blue_h2h = [r["blue_h2h"] for r in results]

    print(f"\n=== Multi-Concept Interference ({N_SEEDS} seeds) ===\n")
    print("learned value for red, single-rule vs. both-rules-active (mean/std):")
    print(f"  single_red condition: {round(statistics.mean(red_single), 2)} / {round(statistics.stdev(red_single), 2)}")
    print(f"  multi condition:      {round(statistics.mean(red_multi), 2)} / {round(statistics.stdev(red_multi), 2)}")
    print("\nlearned value for blue, single-rule vs. both-rules-active (mean/std):")
    print(f"  single_blue condition: {round(statistics.mean(blue_single), 2)} / {round(statistics.stdev(blue_single), 2)}")
    print(f"  multi condition:       {round(statistics.mean(blue_multi), 2)} / {round(statistics.stdev(blue_multi), 2)}")
    print(f"\ngreen (unrelated distractor) value in multi condition, expected ~-1: "
          f"{round(statistics.mean(green_multi), 2)} / {round(statistics.stdev(green_multi), 2)}")

    print(f"\nhead-to-head in multi condition -- picks RED (mean/std): "
          f"{round(statistics.mean(red_h2h), 2)} / {round(statistics.stdev(red_h2h), 2)}")
    print(f"head-to-head in multi condition -- picks BLUE (mean/std): "
          f"{round(statistics.mean(blue_h2h), 2)} / {round(statistics.stdev(blue_h2h), 2)}")

    print("\n--- Pass/fail check ---")
    # Interference test: paired difference (multi - single) should NOT be
    # significantly different from zero, for both red and blue. A single
    # two-sided bootstrap CI on the paired difference answers this directly.
    red_diffs = [m - s for m, s in zip(red_multi, red_single)]
    blue_diffs = [m - s for m, s in zip(blue_multi, blue_single)]
    zero_baseline_red = [0.0] * len(red_diffs)
    zero_baseline_blue = [0.0] * len(blue_diffs)

    rd_diff, rd_lo, rd_hi = bootstrap_diff_ci(red_diffs, zero_baseline_red)
    bd_diff, bd_lo, bd_hi = bootstrap_diff_ci(blue_diffs, zero_baseline_blue)
    red_no_interference = rd_lo <= 0 <= rd_hi
    blue_no_interference = bd_lo <= 0 <= bd_hi

    print(f"red value shifts significantly when blue rule is also active: {not red_no_interference} "
          f"(paired diff mean={rd_diff}, 95% CI=[{rd_lo},{rd_hi}])")
    print(f"blue value shifts significantly when red rule is also active: {not blue_no_interference} "
          f"(paired diff mean={bd_diff}, 95% CI=[{bd_lo},{bd_hi}])")

    behavior_correct = statistics.mean(red_h2h) > statistics.mean(blue_h2h)

    print("\nBuilt-in vs. emergent: the reward rules themselves (red=good, blue=bad)")
    print("are environment ground truth, given the same way every prior experiment's")
    print("hidden rule was given -- never exposed to the agent as labels. Emergent:")
    print("both learned values, and whether the agent's own statistics table keeps")
    print("them independent without cross-contamination -- this is exactly the")
    print("multi-concept interference risk symbolic systems are known to face.")

    if red_no_interference and blue_no_interference and behavior_correct:
        print("\nRESULT: PASS -- the agent holds two independent concepts simultaneously")
        print("without measurable interference between them.")
    else:
        print("\nRESULT: FAIL -- learning one rule measurably corrupted the other, and/or")
        print("resulting behavior does not correctly prefer red over blue. Report exactly")
        print("which condition failed -- this is the multi-concept interference problem")
        print("symbolic systems are known to face, and a real negative result either way.")


if __name__ == "__main__":
    main()
