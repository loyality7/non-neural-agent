"""Entity discovery and causal transfer experiment.

Trains three agents (random baseline, memory-only baseline, and the full
concept-table agent) across many episodes on a fixed training layout, then
evaluates each on a novel red object never encountered during training.
The experiment is repeated across N_SEEDS fixed random seeds and reports
mean and standard deviation, rather than a single run, to distinguish a
genuine effect from sampling noise.
"""

import random
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gridworld import fresh_training_world, RED
from agent.agent import RandomAgent, MemoryOnlyAgent, FullAgent

TRAIN_EPISODES = 200
NOVEL_RED_POS = (7, 0)  # never used in fresh_training_world()
N_SEEDS = 20


def run_episode(agent, world):
    total_gain = 0
    obs = world.observe()
    while world.alive():
        action = agent.act(obs)
        next_obs, delta_energy, done = world.step(action)
        agent.observe_result(obs, action, next_obs, delta_energy)
        total_gain += delta_energy
        obs = next_obs
        if done:
            break
    return total_gain, world.steps


def train(agent_cls, episodes=TRAIN_EPISODES):
    agent = agent_cls()
    survival_times = []
    for _ in range(episodes):
        world = fresh_training_world()
        _, steps = run_episode(agent, world)
        survival_times.append(steps)
    return agent, survival_times


def transfer_test(agent, trials=30, step_budget=80):
    """Place agent far from a brand-new red object (never used in training,
    never eaten by this agent before) and see how often it finds and eats
    it within a step budget.
    """
    successes = 0
    steps_on_success = []
    for _ in range(trials):
        world = fresh_training_world()
        world.objects.clear()
        world.add_object(NOVEL_RED_POS, RED)
        world.agent_pos = (0, 0)
        world.energy = 50
        world.steps = 0

        obs = world.observe()
        for _ in range(step_budget):
            action = agent.act(obs)
            next_obs, delta_energy, done = world.step(action)
            if action == "EAT" and delta_energy > 0:
                successes += 1
                steps_on_success.append(world.steps)
                break
            obs = next_obs
            if done:
                break

    rate = successes / trials
    avg_steps = round(sum(steps_on_success) / len(steps_on_success), 1) if steps_on_success else None
    return rate, avg_steps


def mean_std(values):
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return round(m, 2), round(s, 2)


def run_one_seed(seed):
    random.seed(seed)
    seed_results = {}
    for name, cls in [
        ("random", RandomAgent),
        ("memory_only", MemoryOnlyAgent),
        ("full", FullAgent),
    ]:
        agent, survival = train(cls)
        avg_survival = sum(survival) / len(survival)
        transfer_rate, avg_transfer_steps = transfer_test(agent)
        concept_dump = None
        if hasattr(agent, "concepts"):
            concept_dump = {
                key: round(total / count, 1)
                for key, (count, total) in agent.concepts.stats.items()
                if count >= 3
            }
        seed_results[name] = {
            "avg_survival_steps": avg_survival,
            "transfer_rate": transfer_rate,
            "avg_steps_to_transfer": avg_transfer_steps,
            "concepts": concept_dump,
        }
    return seed_results


def main():
    per_agent = {"random": [], "memory_only": [], "full": []}
    last_concept_dump = {}

    for seed in range(N_SEEDS):
        seed_results = run_one_seed(seed)
        for name, r in seed_results.items():
            per_agent[name].append(r)
            last_concept_dump[name] = r["concepts"]

    print(f"\n=== Entity Discovery and Causal Transfer ({N_SEEDS} seeds) ===\n")
    header = f"{'agent':<14}{'survival(mean/std)':<22}{'transfer_rate(mean/std)':<26}"
    print(header)
    print("-" * len(header))

    summary = {}
    for name, runs in per_agent.items():
        survival_vals = [r["avg_survival_steps"] for r in runs]
        transfer_vals = [r["transfer_rate"] for r in runs]
        surv_m, surv_s = mean_std(survival_vals)
        trans_m, trans_s = mean_std(transfer_vals)
        summary[name] = {"survival": (surv_m, surv_s), "transfer": (trans_m, trans_s)}
        print(f"{name:<14}{f'{surv_m} / {surv_s}':<22}{f'{trans_m} / {trans_s}':<26}")

    print("\nLearned concept table (last seed, feature -> mean delta_energy on EAT):")
    for name in per_agent:
        print(f"  {name}: {last_concept_dump.get(name)}")

    print("\n--- Pass/fail check (evaluated on the mean across seeds) ---")
    full = summary["full"]
    mem = summary["memory_only"]
    rnd = summary["random"]

    passed_transfer = full["transfer"][0] > rnd["transfer"][0] and full["transfer"][0] > mem["transfer"][0]
    passed_vs_memory = full["survival"][0] > mem["survival"][0]
    passed_vs_random = full["survival"][0] > rnd["survival"][0]

    print(f"full agent mean transfer rate beats memory_only and random: {passed_transfer}")
    print(f"full agent mean survival beats memory_only:                 {passed_vs_memory}")
    print(f"full agent mean survival beats random:                      {passed_vs_random}")

    if passed_transfer and passed_vs_memory and passed_vs_random:
        print("\nRESULT: PASS — hypothesis survives this experiment.")
    else:
        print("\nRESULT: FAIL — report honestly, do not reinterpret.")


if __name__ == "__main__":
    main()
