"""Sensitivity sweep for ConceptTable's sliding-window size.

The sliding window's size was chosen as a round number rather than tuned.
This sweep tests whether the re-adaptation result reported for that choice
is fragile to the exact value selected: if a range of window sizes all
show the same qualitative pattern (fast re-adaptation, large effect size),
the round-number choice was not load-bearing. Reuses the causal-flip
experiment's harness, varying only the window size.
"""

import random
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gridworld import fresh_training_world, RED, GREEN
from agent.concepts import ConceptTable
from agent.planner import choose_action
from stats_utils import significant_improvement, cohens_d

PRE_FLIP_EPISODES = 200
POST_FLIP_EPISODES = 200
CHECK_EVERY = 20
NOVEL_GREEN_POS = (7, 0)
N_SEEDS = 20
WINDOW_SIZES = [10, 40, 100]


class WindowedFullAgent:
    def __init__(self, window_size):
        self.concepts = ConceptTable(window_size=window_size)

    def act(self, obs):
        value_fn = lambda f: self.concepts.expected_value(f, "EAT")
        return choose_action(obs["pos"], obs["feature"], obs["visible"], value_fn)

    def observe_result(self, obs, action, next_obs, delta_energy):
        if action == "EAT" and obs["feature"] is not None:
            self.concepts.update(obs["feature"], "EAT", delta_energy)


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


def head_to_head_test(agent, trials=30, step_budget=80):
    green_wins, red_wins = 0, 0
    for _ in range(trials):
        world = fresh_training_world()
        world.objects.clear()
        world.add_object((0, 6), RED)
        world.add_object((6, 0), GREEN)
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
        if outcome == GREEN:
            green_wins += 1
        elif outcome == RED:
            red_wins += 1
    return green_wins / trials, red_wins / trials


def run_one_seed(seed, window_size):
    random.seed(seed)
    agent = WindowedFullAgent(window_size)

    for _ in range(PRE_FLIP_EPISODES):
        run_episode(agent, fresh_training_world(good_color=RED))

    episodes_to_reflect_flip = None
    for i in range(1, POST_FLIP_EPISODES + 1):
        run_episode(agent, fresh_training_world(good_color=GREEN))
        if episodes_to_reflect_flip is None and i % CHECK_EVERY == 0:
            rule, _ = agent.concepts.best_known_rule()
            if rule == (GREEN, "EAT"):
                episodes_to_reflect_flip = i

    green_pref, red_pref = head_to_head_test(agent)
    return episodes_to_reflect_flip, green_pref, red_pref


def main():
    print(f"\n=== Window-Size Sensitivity Sweep, {N_SEEDS} seeds each ===\n")
    print(f"{'window':<10}{'reflip speed (mean/std)':<26}{'green pref':<14}{'red pref':<12}{'Cohen d':<10}")
    print("-" * 72)

    for window in WINDOW_SIZES:
        speeds, green_prefs, red_prefs = [], [], []
        for seed in range(N_SEEDS):
            speed, green, red = run_one_seed(seed, window)
            if speed is not None:
                speeds.append(speed)
            green_prefs.append(green)
            red_prefs.append(red)

        speed_str = f"{round(statistics.mean(speeds), 1)} / {round(statistics.stdev(speeds), 1)}" if len(speeds) > 1 else "never converged"
        never_converged = N_SEEDS - len(speeds)
        d = cohens_d(green_prefs, red_prefs)
        sig, _ = significant_improvement(green_prefs, red_prefs)

        print(f"{window:<10}{speed_str:<26}"
              f"{round(statistics.mean(green_prefs), 2):<14}"
              f"{round(statistics.mean(red_prefs), 2):<12}"
              f"{d:<10}"
              f"{'(sig)' if sig else '(not sig)'}"
              f"{f'  [{never_converged}/{N_SEEDS} never converged]' if never_converged else ''}")

    print("\nInterpretation: if all window sizes show fast re-adaptation (well under")
    print("100 episodes) with a large, significant Cohen's d, the re-adaptation result")
    print("is not fragile to the exact window size chosen.")


if __name__ == "__main__":
    main()
