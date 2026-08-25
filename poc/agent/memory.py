"""Episodic memory: ring buffer of (observation, action, next_observation, delta_energy)."""

from collections import deque


class EpisodicMemory:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def record(self, obs, action, next_obs, delta_energy):
        self.buffer.append((obs, action, next_obs, delta_energy))

    def __len__(self):
        return len(self.buffer)

    def all(self):
        return list(self.buffer)
