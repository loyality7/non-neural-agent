# Non-Neural Developmental Agent

## Overview

A small agent that learns concepts — object categories, conditional rules,
and instrumentally valuable actions — purely through interaction with a
gridworld environment. The architecture contains no neural networks, no
gradient-based optimization, and no large language models at any stage.
Learning is implemented entirely through explicit statistical structures:
sliding-window averages, a learned Markov decision process solved by value
iteration, and count-based novelty tracking.

## Requirements

Python standard library only; no external dependencies. See
`requirements.txt`. Tested on Python 3.14.4; any version >=3.9 should work.

## Running

```
python run.py
```

Runs all experiments in sequence (a few minutes on a laptop CPU). Each can
also be run standalone from the `poc/` directory:

```
python experiments/exp0_entity_discovery.py   # entity grouping and transfer
python experiments/exp1_causal_flip.py        # causal vs. correlation under a rule change
python experiments/exp2_conjunctive_rule.py   # two-condition (conjunctive) rule
python experiments/exp3_planning.py           # value iteration and instrumental value
python experiments/exp4_curiosity.py          # exploration strategy comparison
python experiments/exp5_window_sweep.py       # sliding-window size sensitivity sweep
python experiments/baseline_qlearning.py      # tabular Q-learning baseline
```

Every experiment reports mean and standard deviation across 20 fixed
seeds, along with a bootstrap confidence interval and Cohen's d for any
comparative claim (`stats_utils.py`) — no result is reported from a single
run or an unvalidated percentage difference.

## Layout

```
env/                  environment definitions
  gridworld.py           base gridworld environment
  conjunctive_world.py   adds a two-condition (key + object) causal rule
agent/                 agent architecture
  memory.py               episodic ring buffer
  concepts.py              ConceptTable: sliding-window statistics per (feature, action)
  planner.py               action-selection policies (greedy, count-based curiosity, spatial curiosity)
  value_iteration.py       ConceptGraph: value iteration over a learned two-state MDP
  agent.py                 RandomAgent / MemoryOnlyAgent / FullAgent
experiments/           one script per experiment, each self-contained and runnable
stats_utils.py         bootstrap CI and Cohen's d, used by every experiment's pass/fail check
run.py                 runs all experiments in sequence
```

## Reproducibility

- Every experiment seeds `random.seed(n)` per run for `n` in `range(20)`,
  so re-running reproduces identical results.
- No source of randomness is used outside `random.seed()` -- no wall-clock
  or OS entropy that would cause two runs to diverge.
- Console output is currently the only artifact produced; saving results to
  disk automatically is a known, tracked gap.
