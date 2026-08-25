"""Concept table: a non-parametric, non-gradient statistical model that
aggregates observed outcomes by (feature, action) pair rather than by
object identity.

Grouping by feature instead of by exact observation is the abstraction
mechanism that enables generalization to novel objects sharing a known
feature. Outcomes are tracked in a fixed-size sliding window per key
rather than a lifetime running average, so old evidence ages out and is
fully displaced once the environment's underlying rule changes, instead of
being merely diluted by new evidence.
"""

from collections import defaultdict, deque

WINDOW_SIZE = 40


class ConceptTable:
    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size = window_size
        # (feature, action) -> deque of recent delta_energy outcomes
        self.windows = defaultdict(lambda: deque(maxlen=window_size))

    def update(self, feature, action, delta_energy):
        self.windows[(feature, action)].append(delta_energy)

    def expected_value(self, feature, action):
        window = self.windows.get((feature, action))
        if not window:
            return 0.0
        return sum(window) / len(window)

    def confidence(self, feature, action):
        window = self.windows.get((feature, action))
        return len(window) if window else 0

    def best_known_rule(self):
        """Return the (feature, action) pair with highest expected value, for reporting."""
        best_key, best_val = None, float("-inf")
        for key, window in self.windows.items():
            if len(window) < 3:
                continue
            val = sum(window) / len(window)
            if val > best_val:
                best_key, best_val = key, val
        return best_key, best_val

    @property
    def stats(self):
        """(count, total) view for backward-compat display/dumps -- reflects
        only the current window, not lifetime totals.
        """
        return {key: (len(w), sum(w)) for key, w in self.windows.items()}
