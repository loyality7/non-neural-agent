"""Grounding plumbing: pairs each world object with a symbol, and exposes
that symbol on the observation only when the object is the one currently
being observed/acted on -- not merely visible or nearby.

Four decisions made explicit here, matching the existing project
discipline of stating simplifications rather than letting them hide:

  - Symbol emission is treated as SIMULTANEOUS with the object encounter.
    This is a known simplification -- real language exposure is rarely
    perfectly time-aligned (a word said just before or after seeing the
    thing it names). Handling that misalignment is explicitly out of
    scope here, not silently assumed away.
  - A co-occurrence requires the object to be the CURRENT interaction
    target (the object at the agent's own cell), not merely visible in
    the surrounding radius. This is deliberately strict: a red object
    sitting nearby while a symbol is emitted for a DIFFERENT object the
    agent is actually standing on must not count as a co-occurrence for
    the nearby one. Proximity alone is not exposure.
  - The existing observation record is left completely unmodified in
    shape; only one new optional field is added ("symbol"). Nothing in
    the existing gridworld / concept-graph code path needs to change or
    even be aware this exists -- the same "don't touch what's proven"
    discipline used when Organism was fixed without touching FullAgent.
  - Symbols are attached to OBJECTS only (nouns), not actions/events
    (verbs). Verb-grounding is a known, explicitly flagged next step,
    not something silently out of scope.
"""

from env.gridworld import GridWorld


class GroundingWorld(GridWorld):
    def __init__(self, objects, symbol_map, agent_pos=(0, 0), seed=None, good_color=None, bad_color=None):
        """
        symbol_map: mapping of (x, y) position -> symbol string. Every
        position in `objects` should have a corresponding entry here;
        positions not in objects are ignored.
        """
        super().__init__(objects, agent_pos=agent_pos, seed=seed, good_color=good_color, bad_color=bad_color)
        self.symbol_map = dict(symbol_map)

    def observe(self):
        obs = super().observe()
        # Symbol is exposed ONLY for the object at the agent's own cell --
        # the strict co-occurrence rule. Visible-but-not-occupied objects
        # never carry a symbol, even though their color is still visible.
        obs["symbol"] = self.symbol_map.get(self.agent_pos) if obs["feature"] is not None else None
        return obs
