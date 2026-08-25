"""Habit formation experiment.

Tests whether compiling a repeatedly successful action sequence into a
directly executable routine (agent/habit.py) reduces the number of steps
needed to reach the reward, compared to an agent that continues to
re-derive its plan from currently visible objects every episode.

Two agents are trained on the same conjunctive-rule environment:

  value_iter  -- value iteration over a learned MDP, but action selection
                 still only considers objects currently in sight, falling
                 back to random movement otherwise.
  habit       -- identical value-iteration machinery, plus: once a
                 (feature, key_state) -> position pairing has succeeded
                 repeatedly, navigate directly to that remembered position
                 without waiting for it to become visible.

The environment layout is fixed and deterministic across episodes, so a
remembered position remains valid -- this experiment measures whether the
agent exploits that regularity once it has proven the sequence works, the
way a repeated motor sequence is exploited without re-planning from
scratch each time.
"""

import random
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.conjunctive_world import fresh_conjunctive_world_hard, RED, YELLOW
from agent.value_iteration import ConceptGraph
from agent.habit import HabitAgent
from agent.planner import choose_action
from stats_utils import significant_improvement, cohens_d

from experiments.exp2_conjunctive_rule import run_episode

TRAIN_EPISODES = 600
N_SEEDS = 20
EVAL_TRIALS = 30
EVAL_STEP_BUDGET = 150


class ValueIterationAgent:
    """Reference agent without habit compilation: identical value-iteration
    machinery, action selection limited to currently visible objects.
    """

    def __init__(self, replan_every=10):
        self.graph = ConceptGraph()
        self.replan_every = replan_every
        self._episodes_since_replan = 0

    def act(self, obs):
        key_state = obs["has_key"]
        value_fn = lambda f: self.graph.q_value(f, key_state)
        return choose_action(obs["pos"], obs["feature"], obs["visible"], value_fn, threshold=0.05)

    def observe_result(self, obs, action, next_obs, delta_energy):
        if action == "EAT" and obs["feature"] is not None:
            self.graph.observe(obs["feature"], obs["has_key"], action, delta_energy, next_obs["has_key"])

    def end_of_episode(self):
        self._episodes_since_replan += 1
        if self._episodes_since_replan >= self.replan_every:
            self.graph.run_value_iteration()
            self._episodes_since_replan = 0


def train(agent, episodes=TRAIN_EPISODES):
    survival = []
    for _ in range(episodes):
        world = fresh_conjunctive_world_hard()
        steps = run_episode(agent, world)
        survival.append(steps)
        agent.end_of_episode()
    agent.graph.run_value_iteration()
    return survival


def steps_to_success_test(agent, trials=EVAL_TRIALS, step_budget=EVAL_STEP_BUDGET):
    """Evaluates on the same fixed layout used in training. Records steps
    taken until the agent successfully eats red while holding the key, or
    the step budget is exhausted (recorded as the full budget, a penalty
    for failing to complete the sequence at all).
    """
    steps_taken = []
    successes = 0
    for _ in range(trials):
        world = fresh_conjunctive_world_hard()
        obs = world.observe()
        succeeded = False
        for step_count in range(1, step_budget + 1):
            was_on_red_with_key = obs["feature"] == RED and obs["has_key"]
            action = agent.act(obs)
            next_obs, delta_energy, done = world.step(action)
            if action == "EAT" and was_on_red_with_key and delta_energy > 0:
                steps_taken.append(step_count)
                successes += 1
                succeeded = True
                break
            obs = next_obs
            if done:
                break
        if not succeeded:
            steps_taken.append(step_budget)
    return steps_taken, successes / trials


def run_one_seed(seed):
    random.seed(seed)
    vi_agent = ValueIterationAgent()
    train(vi_agent, TRAIN_EPISODES)

    random.seed(seed)  # identical environment sequence for a fair comparison
    habit_agent = HabitAgent()
    train(habit_agent, TRAIN_EPISODES)

    vi_steps, vi_success_rate = steps_to_success_test(vi_agent)
    habit_steps, habit_success_rate = steps_to_success_test(habit_agent)

    return {
        "vi_mean_steps": statistics.mean(vi_steps),
        "habit_mean_steps": statistics.mean(habit_steps),
        "vi_success_rate": vi_success_rate,
        "habit_success_rate": habit_success_rate,
        "compiled_targets": dict(habit_agent._compiled_targets),
    }


def main():
    results = [run_one_seed(s) for s in range(N_SEEDS)]

    vi_steps = [r["vi_mean_steps"] for r in results]
    habit_steps = [r["habit_mean_steps"] for r in results]
    vi_success = [r["vi_success_rate"] for r in results]
    habit_success = [r["habit_success_rate"] for r in results]

    print(f"\n=== Habit Formation ({N_SEEDS} seeds) ===\n")
    print("mean steps to succeed (lower is better), and success rate within budget:")
    print(f"  value_iter (re-derives plan each time): "
          f"{round(statistics.mean(vi_steps), 1)} steps / "
          f"{round(statistics.mean(vi_success), 2)} success rate")
    print(f"  habit (compiled routine after threshold): "
          f"{round(statistics.mean(habit_steps), 1)} steps / "
          f"{round(statistics.mean(habit_success), 2)} success rate")

    seeds_compiled = sum(1 for r in results if r["compiled_targets"])
    print(f"\nseeds where at least one routine was compiled: {seeds_compiled}/{N_SEEDS}")
    print(f"example compiled targets (seed 0): {results[0]['compiled_targets']}")

    print("\n--- Pass/fail check ---")
    habit_faster, (diff, lo, hi) = significant_improvement(vi_steps, habit_steps)
    d = cohens_d(vi_steps, habit_steps)
    habit_not_worse_success, _ = significant_improvement(habit_success, [s - 0.05 for s in vi_success])

    print(f"habit takes significantly fewer steps than value_iter: {habit_faster} "
          f"(diff={diff}, 95% CI=[{lo},{hi}], Cohen's d={d})")
    print(f"habit success rate mean: {round(statistics.mean(habit_success), 2)}, "
          f"value_iter success rate mean: {round(statistics.mean(vi_success), 2)}")

    print("\nBuilt-in vs. emergent: the compilation trigger (a repetition threshold)")
    print("and the compiled routine's structure (navigate directly to a remembered")
    print("position) are hand-designed. WHICH position gets remembered and WHEN the")
    print("threshold is crossed are both determined purely by the agent's own")
    print("successful experience -- no position is given to it in advance.")

    if habit_faster and statistics.mean(habit_success) >= statistics.mean(vi_success) - 0.05:
        print("\nRESULT: PASS -- compiling a proven sequence into a directly executable")
        print("routine measurably reduces steps to success without harming reliability.")
    else:
        print("\nRESULT: FAIL -- habit compilation did not produce a measurable speed")
        print("advantage, or did so at the cost of reliability.")


if __name__ == "__main__":
    main()
