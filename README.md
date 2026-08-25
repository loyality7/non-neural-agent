# Non-Neural Developmental Agent

A research project testing whether a small agent can learn real concepts —
object categories, conditional rules, and instrumentally valuable actions —
through interaction alone, using no neural networks, no gradient-based
optimization, and no large language models anywhere in the architecture.

Learning is implemented entirely through explicit statistical structures:
sliding-window averages, a learned Markov decision process solved by value
iteration, and count-based novelty tracking. Every experimental claim is
backed by 20-seed runs with bootstrap confidence intervals and Cohen's d,
not single-run numbers or eyeballed percentages.

## Documents

- **[research.md](research.md)** — the full research record: literature
  review, hypothesis, prior-art comparison, and every experiment's results,
  including negative findings and corrections, reported as found.

## Code

All code lives in [`poc/`](poc/) — see [`poc/README.md`](poc/README.md)
for how to run the experiments. Requires only the Python standard library.

```
python poc/run.py
```

## Status

Six experiments completed (entity discovery and transfer, causal-versus-
correlation under a rule change, a two-condition conjunctive rule,
instrumental-value planning via value iteration, and an exploration
strategy comparison), plus a tabular Q-learning baseline and a
hyperparameter sensitivity sweep. Results, including one documented
negative finding, are in `research.md`.
