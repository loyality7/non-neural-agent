"""Sequence-order learning and data-scaling experiment.

Tests two things at once, both pre-registered before this script was run:

1. Can a purely statistical, non-neural sequence memory predict the next
   symbol in a sequence with genuine hidden structure, significantly
   better than an order-blind frequency baseline, using a small amount of
   data (bigram target: within 150 observed transitions)?

2. The harder, more important question: what happens to the data
   requirement when the model needs one more step of context (trigram
   instead of bigram)? Naive expectation is roughly a linear increase
   (~5x, matching the vocabulary size) in transitions needed to reach the
   same accuracy bar. If trigram instead needs a wildly disproportionate
   amount of data (3000-5000+ transitions) to match bigram's accuracy
   level, that is reported as a genuine finding: the symbolic approach has
   its own version of the sparse-data wall that historically pushed
   statistical NLP toward neural methods -- not smoothed over as a partial
   win.

The hidden grammar is order-2 (env/sequence_world.py): the true
next-symbol distribution depends on the two preceding symbols. A bigram
(order-1) model is therefore genuinely misspecified against this grammar,
by design -- this is what makes the trigram-vs-bigram comparison honest,
rather than trigram just re-deriving what bigram could already capture.
"""

import random
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.sequence_world import build_order2_grammar, generate_sequence, VOCAB
from agent.sequence_memory import SequenceMemory
from stats_utils import significant_improvement, cohens_d

N_SEEDS = 20
CHECKPOINTS = [20, 50, 100, 150, 250, 400, 600, 900, 1400, 2000, 3000, 4500, 7000, 10000, 15000, 22000]
TEST_SEQUENCE_LENGTH = 800
BIGRAM_TARGET_TRANSITIONS = 150
ACCEPTABLE_TRIGRAM_SCALING = 750   # ~5x bigram target, naive linear expectation
WALL_TRIGRAM_THRESHOLD = 3000      # pre-registered "this is the wall" line


def accuracy(memory, test_sequence, order):
    correct = 0
    total = 0
    for i in range(order, len(test_sequence)):
        context = tuple(test_sequence[i - order:i])
        prediction = memory.predict(context)
        if prediction == test_sequence[i]:
            correct += 1
        total += 1
    return correct / total


def frequency_baseline_accuracy(memory, test_sequence):
    correct = 0
    total = 0
    guess = memory.predict_frequency_only()
    for symbol in test_sequence[1:]:
        if guess == symbol:
            correct += 1
        total += 1
    return correct / total


def run_one_seed(seed, grammar):
    train_seq = generate_sequence(grammar, length=max(CHECKPOINTS) + 10, order=2, seed=1000 + seed)
    test_seq = generate_sequence(grammar, length=TEST_SEQUENCE_LENGTH, order=2, seed=5000 + seed)

    bigram = SequenceMemory(order=1)
    trigram = SequenceMemory(order=2)

    results_at = {}
    pos = 2  # first two symbols have no order-2 context yet
    for checkpoint in CHECKPOINTS:
        while bigram.total_transitions_observed() < checkpoint and pos < len(train_seq):
            bigram.observe((train_seq[pos - 1],), train_seq[pos])
            trigram.observe((train_seq[pos - 2], train_seq[pos - 1]), train_seq[pos])
            pos += 1
        results_at[checkpoint] = {
            "bigram_acc": accuracy(bigram, test_seq, order=1),
            "trigram_acc": accuracy(trigram, test_seq, order=2),
            "freq_acc": frequency_baseline_accuracy(bigram, test_seq),
        }
    return results_at


def main():
    grammar = build_order2_grammar(VOCAB, seed=0)
    all_results = {c: {"bigram": [], "trigram": [], "freq": []} for c in CHECKPOINTS}

    for seed in range(N_SEEDS):
        seed_results = run_one_seed(seed, grammar)
        for c in CHECKPOINTS:
            all_results[c]["bigram"].append(seed_results[c]["bigram_acc"])
            all_results[c]["trigram"].append(seed_results[c]["trigram_acc"])
            all_results[c]["freq"].append(seed_results[c]["freq_acc"])

    print(f"\n=== Sequence Order Learning & Data Scaling ({N_SEEDS} seeds) ===\n")
    print("Hidden grammar: order-2 (next symbol depends on the two preceding symbols).\n")
    header = f"{'transitions':<14}{'bigram (mean/std)':<22}{'trigram (mean/std)':<22}{'freq baseline':<18}"
    print(header)
    print("-" * len(header))

    # Convergence is defined as reaching 90% of the model's OWN eventual
    # ceiling accuracy (measured at the largest checkpoint), not merely
    # "detectably above the frequency baseline" -- a low-variance but tiny
    # effect (e.g. a genuinely misspecified bigram model against an
    # order-2 grammar) can look "significant" almost immediately without
    # having learned anything close to the real structure. This mirrors a
    # mistake already caught and corrected earlier in this project: a
    # significance check alone is not evidence of having converged to the
    # mechanism's real ceiling.
    bigram_ceiling = statistics.mean(all_results[CHECKPOINTS[-1]]["bigram"])
    trigram_ceiling = statistics.mean(all_results[CHECKPOINTS[-1]]["trigram"])
    bigram_target_acc = bigram_ceiling * 0.9
    trigram_target_acc = trigram_ceiling * 0.9

    bigram_convergence = None
    trigram_convergence = None

    for c in CHECKPOINTS:
        bigram_vals = all_results[c]["bigram"]
        trigram_vals = all_results[c]["trigram"]
        freq_vals = all_results[c]["freq"]

        b_m, b_s = round(statistics.mean(bigram_vals), 3), round(statistics.stdev(bigram_vals), 3)
        t_m, t_s = round(statistics.mean(trigram_vals), 3), round(statistics.stdev(trigram_vals), 3)
        f_m = round(statistics.mean(freq_vals), 3)

        print(f"{c:<14}{f'{b_m} / {b_s}':<22}{f'{t_m} / {t_s}':<22}{f_m:<18}")

        if bigram_convergence is None and b_m >= bigram_target_acc:
            sig, _ = significant_improvement(bigram_vals, freq_vals)
            if sig:
                bigram_convergence = c
        if trigram_convergence is None and t_m >= trigram_target_acc:
            sig, _ = significant_improvement(trigram_vals, freq_vals)
            if sig:
                trigram_convergence = c

    print(f"\n(convergence defined as reaching >=90% of the model's own ceiling accuracy")
    print(f"measured at {CHECKPOINTS[-1]} transitions: bigram ceiling={round(bigram_ceiling,3)}, "
          f"trigram ceiling={round(trigram_ceiling,3)})")

    print("\n--- Pre-registered pass/fail check ---")
    print(f"bigram first beats frequency baseline (significant) at: "
          f"{bigram_convergence if bigram_convergence else 'never, within budget tested'} transitions "
          f"(target: <= {BIGRAM_TARGET_TRANSITIONS})")
    print(f"trigram first beats frequency baseline (significant) at: "
          f"{trigram_convergence if trigram_convergence else 'never, within budget tested'} transitions")
    print(f"acceptable scaling line: <= {ACCEPTABLE_TRIGRAM_SCALING} (~5x bigram target)")
    print(f"pre-registered wall line: >= {WALL_TRIGRAM_THRESHOLD}")

    bigram_passed = bigram_convergence is not None and bigram_convergence <= BIGRAM_TARGET_TRANSITIONS

    if trigram_convergence is None:
        trigram_verdict = "WALL -- never converged within the tested budget"
    elif trigram_convergence <= ACCEPTABLE_TRIGRAM_SCALING:
        trigram_verdict = "ACCEPTABLE -- scales roughly as expected"
    elif trigram_convergence < WALL_TRIGRAM_THRESHOLD:
        trigram_verdict = "DEGRADED -- worse than naive linear scaling, but not the pre-registered wall"
    else:
        trigram_verdict = "WALL -- hit the pre-registered disproportionate-data-need line"

    print(f"\ntrigram scaling verdict: {trigram_verdict}")

    print("\nBuilt-in vs. emergent: the grammar itself (which context implies which")
    print("dominant next symbol) is environment ground truth, never given to either")
    print("model. Emergent: the transition counts, and therefore the predictions and")
    print("the actual number of examples needed to learn the structure -- not assumed,")
    print("measured directly per seed.")

    if bigram_passed and trigram_verdict.startswith("ACCEPTABLE"):
        print("\nRESULT: PASS -- order-sensitive sequence learning works with small data,")
        print("and one additional step of context does not disproportionately explode")
        print("the data requirement.")
    elif bigram_passed:
        print("\nRESULT: PARTIAL -- bigram learning works cheaply as expected, but trigram")
        print(f"scaling is {trigram_verdict.split(' -- ')[0].lower()}. Report exactly this,")
        print("this may be the honest symbolic-approach data wall, not a false negative.")
    else:
        print("\nRESULT: FAIL -- even bigram-level order learning did not converge within")
        print("the pre-registered small-data budget.")


if __name__ == "__main__":
    main()
