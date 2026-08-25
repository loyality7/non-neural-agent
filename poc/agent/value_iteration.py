"""Value iteration over a learned two-state Markov decision process.

Addresses the instrumental-value problem: an object with zero direct
reward but instrumental value (e.g. a key that unlocks a later reward)
will never be sought deliberately by a purely greedy, value-threshold
planner, since its own expected value never clears the threshold on its
own. This module builds a small MDP from experience -- states are a
binary flag (has_key), actions are "eat this feature" -- and runs value
iteration so value from a downstream reward propagates backward onto the
enabling action, even though that action's own immediate reward is zero.

Nothing about which object is instrumental is hand-coded: both the reward
distribution per (feature, state) and the state-transition probability
(does eating this feature tend to flip the state?) are estimated purely
from the agent's own observations.
"""

from collections import defaultdict

GAMMA = 0.9
VI_SWEEPS = 50


class ConceptGraph:
    def __init__(self, window_size=40):
        from collections import deque
        # (feature, key_state, action) -> deque of observed delta_energy
        self.reward_windows = defaultdict(lambda: deque(maxlen=window_size))
        # feature -> deque of observed bools: did eating it (while key=False)
        # flip has_key to True? Only meaningful for the False->? transition;
        # once key=True it's assumed to stay True (matches env design, but
        # the agent verifies this empirically too -- see key_flip_windows).
        self.key_flip_windows = defaultdict(lambda: deque(maxlen=window_size))
        self.values = {False: 0.0, True: 0.0}  # V(has_key) after value iteration

    def observe(self, feature, key_state_before, action, delta_energy, key_state_after):
        if action != "EAT" or feature is None:
            return
        self.reward_windows[(feature, key_state_before)].append(delta_energy)
        if key_state_before is False:
            self.key_flip_windows[feature].append(key_state_after is True)

    def _mean_reward(self, feature, key_state):
        w = self.reward_windows.get((feature, key_state))
        return sum(w) / len(w) if w else 0.0

    def _flip_prob(self, feature):
        w = self.key_flip_windows.get(feature)
        return sum(w) / len(w) if w else 0.0

    def known_features(self):
        return {f for (f, _) in self.reward_windows.keys()}

    def total_count(self, feature):
        """Total samples seen for this feature across both key states.
        Used as a novelty signal (fewest samples = most worth investigating)
        that is independent of any learned value estimate.
        """
        return sum(len(w) for (f, _), w in self.reward_windows.items() if f == feature)

    def run_value_iteration(self, sweeps=VI_SWEEPS, gamma=GAMMA):
        """Classic value iteration over the 2-state MDP: V(s) = max_a Q(s,a).
        Converges in a handful of sweeps for a 2-state chain; run generously
        anyway since it's cheap.
        """
        features = self.known_features()
        V = {False: 0.0, True: 0.0}
        for _ in range(sweeps):
            new_V = {}
            for state in (False, True):
                best_q = 0.0  # staying idle / not eating anything is 0-value fallback
                for feature in features:
                    r = self._mean_reward(feature, state)
                    if state is False:
                        p_flip = self._flip_prob(feature)
                        next_state_value = p_flip * V[True] + (1 - p_flip) * V[False]
                    else:
                        next_state_value = V[True]  # key never lost, by design & by observation
                    q = r + gamma * next_state_value
                    if q > best_q:
                        best_q = q
                new_V[state] = best_q
            V = new_V
        self.values = V
        return V

    def q_value(self, feature, key_state):
        """Q(state, eat-this-feature) using the current value function --
        this is what lets a zero-reward enabler (the key) still score high:
        its Q includes gamma * V(state-after-eating-it).
        """
        r = self._mean_reward(feature, key_state)
        if key_state is False:
            p_flip = self._flip_prob(feature)
            next_state_value = p_flip * self.values[True] + (1 - p_flip) * self.values[False]
        else:
            next_state_value = self.values[True]
        return r + GAMMA * next_state_value
