"""Gridworld variant with a conjunctive (two-condition) causal rule.

Hidden rule (never exposed to the agent): eating a red object yields a
reward only if the agent has already eaten a yellow key earlier in the
same episode; eating red without the key first yields nothing, same as any
other color. Eating the key itself gives no direct reward -- it is a pure
enabler, and its acquisition resets each episode.

This rule cannot be represented by a concept table keyed on a single
feature alone: any such table must either conflate the key-present and
key-absent cases into one diluted estimate, or be extended with a richer
per-context representation.
"""

from env.gridworld import GridWorld, MOVES, ACTIONS, RED, BLUE, GREEN, GRID_SIZE
from env.gridworld import UPKEEP_COST, RED_EAT_BONUS, START_ENERGY

YELLOW = "yellow"
COLORS = [RED, BLUE, GREEN, YELLOW]


class ConjunctiveGridWorld(GridWorld):
    def __init__(self, objects, agent_pos=(0, 0), seed=None):
        super().__init__(objects, agent_pos=agent_pos, seed=seed, good_color=RED)
        self.has_key = False

    def observe(self):
        obs = super().observe()
        obs["has_key"] = self.has_key
        return obs

    def step(self, action):
        assert action in ACTIONS
        self.steps += 1
        delta_energy = -UPKEEP_COST

        if action == "EAT":
            color = self.objects.get(self.agent_pos)
            if color == YELLOW:
                self.has_key = True
            elif color == RED and self.has_key:
                delta_energy = RED_EAT_BONUS - UPKEEP_COST
            if color is not None:
                del self.objects[self.agent_pos]
        else:
            dx, dy = MOVES[action]
            x, y = self.agent_pos
            nx = max(0, min(GRID_SIZE - 1, x + dx))
            ny = max(0, min(GRID_SIZE - 1, y + dy))
            self.agent_pos = (nx, ny)

        self.energy += delta_energy
        obs = self.observe()
        done = self.energy <= 0 or self.steps >= STEP_CAP_LOCAL
        return obs, delta_energy, done

    def alive(self):
        return self.energy > 0 and self.steps < STEP_CAP_LOCAL


STEP_CAP_LOCAL = 300


def fresh_conjunctive_world():
    """Standard layout with the key placed one step from spawn, so locating
    it is trivial regardless of exploration strategy. Isolates whether the
    agent learns the conjunctive rule and its instrumental value from
    whether it can also locate a hard-to-find object -- use
    fresh_conjunctive_world_hard() when exploration strategy itself is
    under test, since this layout provides no signal for that.
    """
    objects = {
        (4, 3): YELLOW,   # key, one step from spawn
        (1, 1): RED,
        (5, 5): RED,
        (0, 7): RED,
        (2, 6): BLUE,
        (6, 2): BLUE,
        (3, 3): GREEN,
    }
    return ConjunctiveGridWorld(objects, agent_pos=(4, 4))


def fresh_conjunctive_world_hard():
    """Variant with the key placed outside the agent's vision radius from
    spawn, so it cannot be located by chance as readily as in
    fresh_conjunctive_world(). Intended for evaluating exploration
    strategies where locating a rare, out-of-sight object is the point.
    Rule and all other objects are identical to fresh_conjunctive_world().
    """
    objects = {
        (0, 0): YELLOW,   # key, far corner -- must be actively searched for
        (1, 1): RED,
        (5, 5): RED,
        (0, 7): RED,
        (2, 6): BLUE,
        (6, 2): BLUE,
        (3, 3): GREEN,
    }
    return ConjunctiveGridWorld(objects, agent_pos=(4, 4))
