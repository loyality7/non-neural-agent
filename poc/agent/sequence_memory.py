"""Sequence memory: tracks what tends to follow what, separate from and
complementary to the concept graph's causal (feature, context) -> outcome
learning.

The concept graph answers "what does this thing do." This answers a
distinct question -- "what tends to come next" -- and is kept as its own
structure rather than folded into the concept graph, the same way direct
value and instrumental value were separated: a data structure trying to
answer two different questions at once is where the earlier value-leak
bug came from.

This is a plain bigram model: a count table of (previous_symbol,
next_symbol) pairs. No gradients, no neural network -- counting and
argmax, the same statistical style as every other mechanism in this
project.
"""

from collections import defaultdict


class SequenceMemory:
    """context (a tuple of `order` preceding symbols) -> next-symbol counts.
    order=1 is a plain bigram model; order=2 is trigram (conditions on the
    two preceding symbols), etc. Same counting mechanism regardless of
    order -- only the context key's length changes.
    """

    def __init__(self, order=1):
        self.order = order
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.symbol_counts = defaultdict(int)  # for the frequency-only baseline

    def observe(self, context, next_symbol):
        """context: a tuple of exactly `self.order` preceding symbols."""
        self.transition_counts[context][next_symbol] += 1
        self.symbol_counts[next_symbol] += 1

    def predict(self, context):
        """Best guess for the symbol following `context`, using what has
        actually followed this exact context before. Falls back to the
        globally most frequent symbol if this context has never been seen.
        """
        counts = self.transition_counts.get(context)
        if not counts:
            return self.predict_frequency_only()
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def predict_frequency_only(self):
        """The order-blind baseline: ignores context entirely, always
        guesses whichever symbol has been most common overall.
        """
        if not self.symbol_counts:
            return None
        return max(self.symbol_counts.items(), key=lambda kv: kv[1])[0]

    def total_transitions_observed(self):
        return sum(self.symbol_counts.values())
