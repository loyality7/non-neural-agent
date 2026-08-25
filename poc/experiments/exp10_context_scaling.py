"""Context-size scaling experiment: order-1 vs. order-2 vs. order-3,
same vocabulary, directly comparable convergence metric.

exp9 found a real effect (bigram/trigram both needed more data than
hoped) but its convergence metric was entangled with a significance test
against a noisy baseline, whose sensitivity differs depending on how
close a model's ceiling is to chance -- making the two numbers not
directly comparable as points on a scaling curve. This experiment fixes
that: convergence here means "reached within 5% (relative) of the
model's OWN final-checkpoint accuracy," which is comparable across orders
regardless of how high or low each order's ceiling happens to be.

Pre-registered target, written down before this was run (per the
discipline used throughout this project):

  Naive n-gram theory says the number of transitions needed to converge
  should scale roughly with the number of distinct contexts, which is
  vocab_size^order. Going from order N to order N+1 multiplies the
  context space by vocab_size, so the "expected, survivable" (FRICTION)
  case is: transitions-to-converge also grows by roughly that same
  factor, order over order, give or take ~3x for noise. The WALL case is:
  the ratio grows much faster than that, especially if it visibly
  ACCELERATES from order 1->2 to order 2->3 (super-linear / combinatorial
  blowup, the historically documented reason n-grams were abandoned for
  neural sequence models in the 2010s). If order-3 fails to converge at
  all within a generously large budget, that's the wall confirmed outright.
"""

import random
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.sequence_world import build_grammar, generate_sequence
from agent.sequence_memory import SequenceMemory

VOCAB = ["A", "B", "C", "D", "E", "F"]
DOMINANT_PROB = 0.5
N_SEEDS = 15  # reduced from 20 for runtime given order-3's much larger budget sweep
TEST_SEQUENCE_LENGTH = 800
CONVERGENCE_FRACTION = 0.95  # "converged" = within 95% of its own final-checkpoint accuracy

CHECKPOINTS_BY_ORDER = {
    1: [20, 50, 100, 200, 400, 700, 1000, 1500, 2200],
    2: [50, 150, 400, 900, 1800, 3200, 5000, 7500, 11000, 16000],
    3: [200, 600, 1500, 3500, 7000, 13000, 22000, 35000, 55000, 80000, 110000],
}


def accuracy(memory, test_sequence, order):
    correct = 0
    total = 0
    for i in range(order, len(test_sequence)):
        context = tuple(test_sequence[i - order:i])
        if memory.predict(context) == test_sequence[i]:
            correct += 1
        total += 1
    return correct / total


def run_one_seed_one_order(order, seed, grammar, checkpoints):
    max_len = max(checkpoints) + order + 10
    train_seq = generate_sequence(grammar, length=max_len, order=order, vocab=VOCAB, seed=1000 + seed)
    test_seq = generate_sequence(grammar, length=TEST_SEQUENCE_LENGTH, order=order, vocab=VOCAB, seed=5000 + seed)

    memory = SequenceMemory(order=order)
    results_at = {}
    pos = order
    for checkpoint in checkpoints:
        while memory.total_transitions_observed() < checkpoint and pos < len(train_seq):
            context = tuple(train_seq[pos - order:pos])
            memory.observe(context, train_seq[pos])
            pos += 1
        results_at[checkpoint] = accuracy(memory, test_seq, order)
    return results_at


def find_convergence_point(mean_accuracy_by_checkpoint, checkpoints):
    final_acc = mean_accuracy_by_checkpoint[checkpoints[-1]]
    target = final_acc * CONVERGENCE_FRACTION
    for c in checkpoints:
        if mean_accuracy_by_checkpoint[c] >= target:
            return c, final_acc
    return None, final_acc


def main():
    print(f"\n=== Context-Size Scaling: Order-1 vs. Order-2 vs. Order-3 ({N_SEEDS} seeds) ===\n")
    print(f"vocabulary size: {len(VOCAB)}, dominant probability: {DOMINANT_PROB}")
    print(f"convergence = reaching >= {int(CONVERGENCE_FRACTION*100)}% of the model's own final accuracy\n")

    convergence_points = {}
    ceilings = {}

    for order in (1, 2, 3):
        checkpoints = CHECKPOINTS_BY_ORDER[order]
        grammar = build_grammar(order, VOCAB, seed=0, dominant_prob=DOMINANT_PROB)

        all_by_checkpoint = {c: [] for c in checkpoints}
        for seed in range(N_SEEDS):
            seed_results = run_one_seed_one_order(order, seed, grammar, checkpoints)
            for c in checkpoints:
                all_by_checkpoint[c].append(seed_results[c])

        mean_by_checkpoint = {c: statistics.mean(vals) for c, vals in all_by_checkpoint.items()}
        conv_point, final_acc = find_convergence_point(mean_by_checkpoint, checkpoints)

        convergence_points[order] = conv_point
        ceilings[order] = final_acc

        print(f"order-{order} (context space = {len(VOCAB)**order} possible contexts):")
        for c in checkpoints:
            print(f"  {c:<8} accuracy={round(mean_by_checkpoint[c], 3)}")
        print(f"  -> converged at: {conv_point if conv_point else 'NOT within tested budget'} transitions, "
              f"final accuracy={round(final_acc, 3)}\n")

    print("--- Pre-registered scaling check ---")
    if all(convergence_points[o] is not None for o in (1, 2, 3)):
        ratio_2_1 = convergence_points[2] / convergence_points[1]
        ratio_3_2 = convergence_points[3] / convergence_points[2]
        print(f"order-1 -> order-2 convergence ratio: {round(ratio_2_1, 2)}x (naive expectation: ~{len(VOCAB)}x)")
        print(f"order-2 -> order-3 convergence ratio: {round(ratio_3_2, 2)}x (naive expectation: ~{len(VOCAB)}x)")

        friction = ratio_2_1 <= len(VOCAB) * 3 and ratio_3_2 <= len(VOCAB) * 3
        accelerating = ratio_3_2 > ratio_2_1 * 1.5

        if friction and not accelerating:
            print("\nRESULT: FRICTION -- data need grows roughly in line with context-space size,")
            print("order over order. Annoying (more data needed at each step) but survivable --")
            print("not the combinatorial wall. Non-neural fixes (smoothing, backoff) are the")
            print("appropriate next tool if this needs to scale further, not a neural network.")
        elif accelerating:
            print("\nRESULT: WALL -- the ratio is ACCELERATING from order 1->2 to order 2->3,")
            print("not just growing. This is the real, historically-documented signature of the")
            print("problem that pushed statistical NLP toward neural methods in the 2010s.")
            print("Report exactly this. Non-neural mitigations (Kneser-Ney smoothing, Bayesian")
            print("backoff/hierarchical priors) exist and are the honest next architectural")
            print("piece to investigate -- not evidence pure counting is dead, but evidence")
            print("raw unsmoothed counting alone will not scale past small context sizes.")
        else:
            print("\nRESULT: DEGRADED -- worse than naive linear scaling but not clearly")
            print("accelerating. Genuinely ambiguous with only two ratios; would need order-4")
            print("to tell friction from an early wall with confidence.")
    else:
        missing = [o for o in (1, 2, 3) if convergence_points[o] is None]
        print(f"order(s) {missing} did NOT converge within the tested budget.")
        print("\nRESULT: WALL -- confirmed outright. At least one order failed to reach its own")
        print("stable accuracy even with a generously large transition budget.")


if __name__ == "__main__":
    main()
