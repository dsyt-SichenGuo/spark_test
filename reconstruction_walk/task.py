"""Keyboard intent and simple local fall recovery, independent of Isaac imports."""

import math
import numpy as np


class KeyboardWalkTask:
    def __init__(self, *, limits=(.12, .08, .30), dt=.02, fall_confirm_steps=8):
        self.limits = np.array(limits, dtype=float)
        self.dt = dt
        self.fall_confirm_steps = fall_confirm_steps
        self.keys = set()
        self.command = np.zeros(3)
        self.age = self.fall_count = 0
        self.reset_requested = False
        self.last_safe = None

    def clear_input(self):
        self.keys.clear()
        self.command[:] = 0

    def reset(self, position=None):
        self.clear_input()
        self.age = self.fall_count = 0
        self.reset_requested = False
        if position is not None:
            self.last_safe = np.asarray(position, dtype=float).copy()

    def compute_command(self):
        pressed = lambda a, b: float(a in self.keys) - float(b in self.keys)
        target = self.limits * np.array([pressed("W", "S"), pressed("Q", "E"), pressed("A", "D")])
        if "SPACE" in self.keys or self.age < 70:
            target[:] = 0
        # Per-second acceleration limits; the control period is explicit.
        acceleration = np.array([.3, .25, .8]) * self.dt
        self.command += np.clip(target - self.command, -acceleration, acceleration)
        return self.command.copy()

    def observe(self, root_xyzw, floor_height):
        self.age += 1
        root = np.asarray(root_xyzw)
        if not np.isfinite(root).all():
            return "nonfinite"
        up = 1 - 2 * (root[3] ** 2 + root[4] ** 2)
        low = math.isfinite(floor_height) and root[2] - floor_height < .30
        bad = low or up < math.cos(math.radians(60))
        self.fall_count = self.fall_count + 1 if bad else 0
        if self.reset_requested:
            return "manual"
        if self.fall_count >= self.fall_confirm_steps:
            return "fallen"
        if root[2] < -2:
            return "below_environment"
        if not bad and self.age > 100 and math.isfinite(floor_height) and up > .95:
            self.last_safe = root[:3].copy()
        return None

    def choose_spawn(self, candidates, current_position, radius=2.):
        candidates = np.asarray(candidates)
        current = np.asarray(current_position)
        if np.isfinite(current[:2]).all():
            distances = np.linalg.norm(candidates[:, :2] - current[:2], axis=1)
            nearby = np.flatnonzero(distances <= radius)
            if len(nearby):
                return candidates[nearby[np.argmin(distances[nearby])]].copy()
        if self.last_safe is not None:
            distances = np.linalg.norm(candidates[:, :2] - self.last_safe[:2], axis=1)
            return candidates[np.argmin(distances)].copy()
        return candidates[0].copy()
