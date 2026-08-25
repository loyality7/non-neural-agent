"""Synthetic sequence world with a genuine order-2 (trigram) hidden
grammar: the true next-symbol distribution depends on the two preceding
symbols, not just one. This is deliberate -- a grammar with only order-1
structure would let a bigram model fully capture it, which would not test
anything about the cost of adding context length. An order-2 grammar
forces a bigram (order-1) model to be genuinely misspecified, and lets a
trigram (order-2) model's real data requirement be measured honestly
against a bigram's.

The transition table is a fixed, seeded random assignment: every
(prev2, prev1) pair has one dominant next symbol (probability 0.75) and
the remaining probability split uniformly over the other symbols. This
gives real, learnable structure -- not noise -- while keeping the
generator itself simple and auditable.
"""

import random

VOCAB = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
DOMINANT_PROB = 0.45


def build_grammar(order, vocab=VOCAB, seed=0, dominant_prob=DOMINANT_PROB):
    """Returns a dict {context_tuple: {symbol: probability}} covering
    every possible context of length `order`. Generalizes
    build_order2_grammar to arbitrary order, so order-1/2/3 grammars are
    all built the same way and are directly comparable.
    """
    import itertools

    rng = random.Random(seed)
    grammar = {}
    for context in itertools.product(vocab, repeat=order):
        dominant = rng.choice(vocab)
        others = [s for s in vocab if s != dominant]
        remainder = 1.0 - dominant_prob
        probs = {dominant: dominant_prob}
        for s in others:
            probs[s] = remainder / len(others)
        grammar[context] = probs
    return grammar


def build_order2_grammar(vocab=VOCAB, seed=0, dominant_prob=DOMINANT_PROB):
    return build_grammar(2, vocab=vocab, seed=seed, dominant_prob=dominant_prob)


def sample_next(grammar, context, rng):
    probs = grammar[context]
    symbols = list(probs.keys())
    weights = list(probs.values())
    return rng.choices(symbols, weights=weights, k=1)[0]


def generate_sequence(grammar, length, order, vocab=VOCAB, seed=0):
    """Generates one sequence of `length` symbols. The first `order`
    symbols are drawn uniformly at random (no history yet to condition
    on); every symbol after that follows the grammar.
    """
    rng = random.Random(seed)
    seq = [rng.choice(vocab) for _ in range(order)]
    for _ in range(length - order):
        context = tuple(seq[-order:])
        next_symbol = sample_next(grammar, context, rng)
        seq.append(next_symbol)
    return seq
