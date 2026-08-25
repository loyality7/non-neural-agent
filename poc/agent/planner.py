"""Action-selection policies for the gridworld agent.

Each policy chooses a movement or EAT action given the agent's current
position, the feature at its own cell, the set of currently visible
objects, and a value function learned from past experience. Three
variants are provided, of increasing exploration sophistication:

- choose_action: greedy toward the best-known visible object, random
  movement otherwise.
- choose_action_curious: adds a count-based novelty fallback among
  visible objects when nothing clears the value threshold.
- choose_action_spatial_curious: adds a second, spatial novelty
  fallback -- prefer the least-visited neighboring cell -- for when no
  object is currently visible at all.
"""

import random

from env.gridworld import MOVES, GRID_SIZE


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_toward(pos, target):
    """One grid step reducing Manhattan distance to target."""
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    candidates = []
    if dx > 0:
        candidates.append("E")
    elif dx < 0:
        candidates.append("W")
    if dy > 0:
        candidates.append("S")
    elif dy < 0:
        candidates.append("N")
    if not candidates:
        return random.choice(list(MOVES.keys()))
    return random.choice(candidates)


def choose_action(pos, current_feature, visible_objects, value_fn, threshold=1.0):
    """Greedy action selection: eat the current object if standing on one;
    otherwise walk toward the best-valued visible object above `threshold`;
    otherwise move randomly.

    visible_objects: mapping of position to feature for everything
        currently in sight. Visibility only reveals a feature's identity,
        not its value.
    value_fn: callable mapping a feature to its expected outcome, as
        learned from past experience with that feature.
    """
    if current_feature is not None:
        return "EAT"

    best_target, best_val = None, threshold
    for opos, feature in visible_objects.items():
        val = value_fn(feature)
        if val > best_val:
            best_target, best_val = opos, val

    if best_target is not None and best_target != pos:
        return step_toward(pos, best_target)

    return random.choice(list(MOVES.keys()))


def choose_action_curious(pos, current_feature, visible_objects, value_fn, count_fn, threshold=1.0):
    """Same as choose_action, with a count-based novelty fallback: when
    nothing visible clears the value threshold, walk toward whichever
    visible object has been sampled the fewest times rather than moving
    randomly. This targets objects whose value is not yet well estimated,
    which pure random exploration can systematically under-sample.

    count_fn: callable mapping a feature to the number of times it has
        been sampled so far.
    """
    if current_feature is not None:
        return "EAT"

    best_target, best_val = None, threshold
    for opos, feature in visible_objects.items():
        val = value_fn(feature)
        if val > best_val:
            best_target, best_val = opos, val

    if best_target is not None and best_target != pos:
        return step_toward(pos, best_target)

    if visible_objects:
        least_known_target = min(visible_objects.items(), key=lambda kv: count_fn(kv[1]))[0]
        if least_known_target != pos:
            return step_toward(pos, least_known_target)

    return random.choice(list(MOVES.keys()))


def choose_action_spatial_curious(pos, current_feature, visible_objects, value_fn,
                                   object_count_fn, cell_count_fn, threshold=1.0):
    """Extends choose_action_curious with a second fallback layer: object
    novelty only applies to objects currently in sight, so it provides no
    guidance once nothing of interest is visible. This adds a spatial
    novelty signal -- when no visible object is worth walking to, move
    toward whichever adjacent cell has been visited the fewest times.

    cell_count_fn: callable mapping a grid position to the number of times
        the agent has occupied it.
    """
    if current_feature is not None:
        return "EAT"

    best_target, best_val = None, threshold
    for opos, feature in visible_objects.items():
        val = value_fn(feature)
        if val > best_val:
            best_target, best_val = opos, val
    if best_target is not None and best_target != pos:
        return step_toward(pos, best_target)

    if visible_objects:
        least_known_target = min(visible_objects.items(), key=lambda kv: object_count_fn(kv[1]))[0]
        if least_known_target != pos:
            return step_toward(pos, least_known_target)

    # Nothing visible worth acting on -- fall back to spatial novelty:
    # move toward whichever neighboring cell has been visited least.
    # Clamp candidates to the grid the same way GridWorld.step() does --
    # an unclamped off-grid coordinate always reads as "unvisited" (count 0)
    # and can never actually be reached, which traps the agent forever
    # wanting to step off the board near any edge.
    x, y = pos
    candidates = []
    for move, (dx, dy) in MOVES.items():
        next_pos = (
            max(0, min(GRID_SIZE - 1, x + dx)),
            max(0, min(GRID_SIZE - 1, y + dy)),
        )
        candidates.append((cell_count_fn(next_pos), move))
    min_count = min(c for c, _ in candidates)
    least_visited_moves = [m for c, m in candidates if c == min_count]
    return random.choice(least_visited_moves)
