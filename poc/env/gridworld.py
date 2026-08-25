"""Minimal gridworld environment for testing entity discovery and causal
transfer in a non-gradient agent.

The environment enforces a single hidden rule that is never exposed to the
agent: eating an object of `good_color` yields a large energy reward; every
other color/action combination only costs the per-step upkeep. The agent
must discover this rule purely from its own interaction history.
"""

GRID_SIZE = 8
MOVES = {
    "N": (0, -1),
    "S": (0, 1),
    "E": (1, 0),
    "W": (-1, 0),
}
ACTIONS = list(MOVES.keys()) + ["EAT"]

RED = "red"
BLUE = "blue"
GREEN = "green"
COLORS = [RED, BLUE, GREEN]

UPKEEP_COST = 1
RED_EAT_BONUS = 20
BAD_COLOR_PENALTY = 20
START_ENERGY = 50
STEP_CAP = 300


class GridWorld:
    def __init__(self, objects, agent_pos=(0, 0), seed=None, good_color=RED, bad_color=None):
        """
        objects: mapping of (x, y) grid position to color string.
        good_color: the color that currently yields a reward on EAT. Kept
            as a parameter rather than a constant so the reward rule can be
            changed at runtime without modifying this class.
        bad_color: an optional second, independent color that yields an
            extra penalty on EAT, unrelated to good_color. Used to test
            whether an agent can hold two independent rules at once
            without them interfering in its learned statistics.
        """
        self.objects = dict(objects)
        self.agent_pos = agent_pos
        self.energy = START_ENERGY
        self.steps = 0
        self.good_color = good_color
        self.bad_color = bad_color

    VISION_RADIUS = 2

    def observe(self):
        """Return the agent's current observation: its position, the color
        at its own cell (or None), and every object visible within
        VISION_RADIUS (Chebyshev distance). Visibility only reveals an
        object's color, not its value -- the agent learns an object's
        outcome only by standing on it and eating it.
        """
        color = self.objects.get(self.agent_pos)
        ax, ay = self.agent_pos
        visible = {}
        for (ox, oy), c in self.objects.items():
            if max(abs(ox - ax), abs(oy - ay)) <= self.VISION_RADIUS:
                visible[(ox, oy)] = c
        return {"pos": self.agent_pos, "feature": color, "visible": visible}

    def step(self, action):
        assert action in ACTIONS
        self.steps += 1
        delta_energy = -UPKEEP_COST

        if action == "EAT":
            color = self.objects.get(self.agent_pos)
            if color == self.good_color:
                delta_energy = RED_EAT_BONUS - UPKEEP_COST
            elif self.bad_color is not None and color == self.bad_color:
                delta_energy = -BAD_COLOR_PENALTY - UPKEEP_COST
            # eating removes the object regardless of color
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
        done = self.energy <= 0 or self.steps >= STEP_CAP
        return obs, delta_energy, done

    def add_object(self, pos, color):
        self.objects[pos] = color

    def alive(self):
        return self.energy > 0 and self.steps < STEP_CAP


def fresh_training_world(good_color=RED, bad_color=None):
    """Standard training layout: a few of each color scattered around."""
    objects = {
        (1, 1): RED,
        (5, 5): RED,
        (2, 6): BLUE,
        (6, 2): BLUE,
        (3, 3): GREEN,
        (0, 7): GREEN,
    }
    return GridWorld(objects, agent_pos=(4, 4), good_color=good_color, bad_color=bad_color)
