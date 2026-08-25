"""Causal-versus-correlation experiment: mid-run rule reversal.

The environment's hidden rule changes mid-run: red is rewarding for the
first PRE_FLIP_EPISODES episodes, then green becomes rewarding for
POST_FLIP_EPISODES episodes while red reverts to neutral. The same agent
and concept table are used throughout, with no reset at the flip point.

This tests whether the agent detects the change and re-learns -- adopting
green as its new best rule and transferring that preference to a novel
green object -- or remains anchored to its earlier red-is-good belief.
"""

import random
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gridworld import fresh_training_world, RED, GREEN
from agent.agent import FullAgent
from stats_utils import significant_improvement, cohens_d

PRE_FLIP_EPISODES = 200
POST_FLIP_EPISODES = 200
CHECK_EVERY = 20  # episodes, for tracking re-adaptation speed
NOVEL_GREEN_POS = (7, 0)
N_SEEDS = 20


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


def transfer_test_color(agent, color, trials=30, step_budget=80):
    """Behavioral preference test: does the agent choose to seek out and eat
    a novel object of the given color? The test world's reward rule is
    irrelevant here by design -- this measures the agent's learned belief,
    not the environment's current ground truth.
    """
    successes = 0
    for _ in range(trials):
        world = fresh_training_world()
        world.objects.clear()
        world.add_object(NOVEL_GREEN_POS, color)
        world.agent_pos = (0, 0)
        world.energy = 50
        world.steps = 0

        obs = world.observe()
        for _ in range(step_budget):
            was_on_object = obs["feature"] == color
            action = agent.act(obs)
            next_obs, delta_energy, done = world.step(action)
            if action == "EAT" and was_on_object:
                successes += 1
                break
            obs = next_obs
            if done:
                break
    return successes / trials


def head_to_head_test(agent, trials=30, step_budget=80):
    """Place a novel red object and a novel green object equidistant from
    the agent. Whichever it eats first reflects its actual current
    preference between the two -- a direct test of relative ranking,
    rather than of whether it will act on either color at all.
    """
    green_wins = 0
    red_wins = 0
    neither = 0
    for _ in range(trials):
        world = fresh_training_world()
        world.objects.clear()
        world.add_object((0, 6), RED)
        world.add_object((6, 0), GREEN)
        world.agent_pos = (3, 3)  # equidistant (manhattan 6 from both)
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
        if outcome == GREEN:
            green_wins += 1
        elif outcome == RED:
            red_wins += 1
        else:
            neither += 1
    return green_wins / trials, red_wins / trials, neither / trials


def run_one_seed(seed):
    random.seed(seed)
    agent = FullAgent()

    # First phase: red is the rewarding color.
    for _ in range(PRE_FLIP_EPISODES):
        run_episode(agent, fresh_training_world(good_color=RED))

    pre_flip_best_rule, _ = agent.concepts.best_known_rule()
    pre_flip_transfer_red = transfer_test_color(agent, RED)

    # Second phase: the rule flips -- green becomes rewarding, red neutral.
    episodes_to_reflect_flip = None
    for i in range(1, POST_FLIP_EPISODES + 1):
        run_episode(agent, fresh_training_world(good_color=GREEN))
        if episodes_to_reflect_flip is None and i % CHECK_EVERY == 0:
            rule, _ = agent.concepts.best_known_rule()
            if rule == (GREEN, "EAT"):
                episodes_to_reflect_flip = i

    post_flip_best_rule, _ = agent.concepts.best_known_rule()
    post_flip_transfer_green = transfer_test_color(agent, GREEN)
    post_flip_transfer_red = transfer_test_color(agent, RED)
    green_pref, red_pref, neither_pref = head_to_head_test(agent)

    return {
        "pre_flip_best_rule": pre_flip_best_rule,
        "pre_flip_transfer_red": pre_flip_transfer_red,
        "post_flip_best_rule": post_flip_best_rule,
        "episodes_to_reflect_flip": episodes_to_reflect_flip,
        "post_flip_transfer_green": post_flip_transfer_green,
        "post_flip_transfer_red": post_flip_transfer_red,
        "head_to_head_green": green_pref,
        "head_to_head_red": red_pref,
        "head_to_head_neither": neither_pref,
        "final_concepts": dict(agent.concepts.stats),
    }


def main():
    results = [run_one_seed(s) for s in range(N_SEEDS)]

    re_adapted = [r for r in results if r["post_flip_best_rule"] == (GREEN, "EAT")]
    adapt_speeds = [r["episodes_to_reflect_flip"] for r in re_adapted if r["episodes_to_reflect_flip"]]

    green_transfer = [r["post_flip_transfer_green"] for r in results]
    red_transfer_after = [r["post_flip_transfer_red"] for r in results]

    print(f"\n=== Causal vs. Correlation: Mid-Run Rule Reversal ({N_SEEDS} seeds) ===\n")
    print(f"seeds where best rule correctly flipped red->green: {len(re_adapted)}/{N_SEEDS}")
    if adapt_speeds:
        print(f"episodes needed to re-adapt (mean/std): "
              f"{round(statistics.mean(adapt_speeds), 1)} / "
              f"{round(statistics.stdev(adapt_speeds), 1) if len(adapt_speeds) > 1 else 0.0}")
    else:
        print("episodes needed to re-adapt: never re-adapted within budget in ANY seed")

    print(f"post-flip transfer rate to novel GREEN (mean/std): "
          f"{round(statistics.mean(green_transfer), 2)} / "
          f"{round(statistics.stdev(green_transfer), 2) if len(green_transfer) > 1 else 0.0}")
    print(f"post-flip transfer rate to RED, should have dropped (mean/std): "
          f"{round(statistics.mean(red_transfer_after), 2)} / "
          f"{round(statistics.stdev(red_transfer_after), 2) if len(red_transfer_after) > 1 else 0.0}")

    green_h2h = [r["head_to_head_green"] for r in results]
    red_h2h = [r["head_to_head_red"] for r in results]
    print(f"\nhead-to-head (both visible, equidistant) -- picks GREEN (mean/std): "
          f"{round(statistics.mean(green_h2h), 2)} / "
          f"{round(statistics.stdev(green_h2h), 2) if len(green_h2h) > 1 else 0.0}")
    print(f"head-to-head -- picks RED (mean/std): "
          f"{round(statistics.mean(red_h2h), 2)} / "
          f"{round(statistics.stdev(red_h2h), 2) if len(red_h2h) > 1 else 0.0}")

    print("\nExample final concept table (seed 0):")
    print(f"  {results[0]['final_concepts']}")

    print("\n--- Pass/fail check ---")
    passed_reflip = len(re_adapted) >= N_SEEDS * 0.8  # 80% of seeds must re-adapt

    is_significant, (diff_mean, ci_lo, ci_hi) = significant_improvement(green_h2h, red_h2h)
    effect_size = cohens_d(green_h2h, red_h2h)

    print(f"agent re-learns green as best rule in >=80% of seeds: {passed_reflip}")
    print(f"green - red head-to-head diff: {diff_mean} (95% bootstrap CI: [{ci_lo}, {ci_hi}])")
    print(f"CI excludes zero (i.e. not just sampling noise across these {N_SEEDS} seeds): {is_significant}")
    print(f"effect size (Cohen's d): {effect_size} "
          f"({'negligible' if abs(effect_size) < 0.2 else 'small' if abs(effect_size) < 0.5 else 'medium' if abs(effect_size) < 0.8 else 'large'})")

    if passed_reflip and is_significant:
        verdict = "PASS"
        if abs(effect_size) < 0.5:
            verdict += " (statistically significant but small effect size)"
        print(f"\nRESULT: {verdict} -- agent detects the causal change and re-learns.")
    else:
        print("\nRESULT: FAIL -- agent is stuck on its prior belief, re-adapts too slowly or")
        print("unreliably, or the apparent green-over-red preference does not survive a")
        print("bootstrap significance check.")


if __name__ == "__main__":
    main()
