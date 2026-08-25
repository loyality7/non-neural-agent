"""Noisy-perception variant of the gridworld.

Every prior experiment gave the agent a clean color string as its
observation -- "red", "blue", "green" -- with no perceptual work required
to use it. This environment replaces that with a noisy multi-attribute
numeric signal: each true color has a fixed underlying attribute vector,
and every observation of it is that vector plus independent Gaussian
noise. The agent never sees the color string at all. Before any of the
existing causal-learning machinery can apply, the agent must first turn
these noisy vectors into a small number of stable, reusable categories on
its own -- the first genuine perceptual-discovery step in this project.
"""

import random

from env.gridworld import GridWorld, GRID_SIZE, RED, BLUE, GREEN, UPKEEP_COST, RED_EAT_BONUS, ACTIONS, MOVES

# Fixed "true" attribute vectors per color, unknown to the agent. Chosen
# far enough apart that a reasonable clustering method can separate them
# under noise, close enough that the task isn't trivial.
TRUE_ATTRIBUTES = {
    RED: (0.9, 0.1, 0.2),
    BLUE: (0.1, 0.1, 0.9),
    GREEN: (0.1, 0.9, 0.1),
}

NOISE_STDDEV = 0.12


class NoisyPerceptionWorld(GridWorld):
    def __init__(self, objects, agent_pos=(0, 0), seed=None, good_color=RED, noise_stddev=NOISE_STDDEV):
        super().__init__(objects, agent_pos=agent_pos, seed=seed, good_color=good_color)
        self.noise_stddev = noise_stddev

    def _noisy_vector(self, color):
        true_vec = TRUE_ATTRIBUTES[color]
        return tuple(v + random.gauss(0, self.noise_stddev) for v in true_vec)

    def observe(self):
        color = self.objects.get(self.agent_pos)
        ax, ay = self.agent_pos
        visible = {}
        for (ox, oy), c in self.objects.items():
            if max(abs(ox - ax), abs(oy - ay)) <= self.VISION_RADIUS:
                visible[(ox, oy)] = self._noisy_vector(c)
        feature = self._noisy_vector(color) if color is not None else None
        return {"pos": self.agent_pos, "feature": feature, "visible": visible}


def fresh_perception_world(good_color=RED, noise_stddev=NOISE_STDDEV):
    objects = {
        (1, 1): RED,
        (5, 5): RED,
        (2, 6): BLUE,
        (6, 2): BLUE,
        (3, 3): GREEN,
        (0, 7): GREEN,
    }
    return NoisyPerceptionWorld(objects, agent_pos=(4, 4), good_color=good_color, noise_stddev=noise_stddev)
