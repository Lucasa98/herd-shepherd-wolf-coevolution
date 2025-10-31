import numpy as np
import pygame
from models.sheep import Sheep
from models.shepherd import Shepherd


class World:
    def __init__(self, width, height, params, rng: np.random.Generator):
        self.width = width
        self.height = height
        self.rng = rng

        self.entities = []

        # Ovejas
        N = 50
        rand_positions = rng.uniform(0, 1, size=(N, 2))
        rand_positions[:, 0] *= self.width
        rand_positions[:, 1] *= self.height
        self.ovejas = [Sheep(rand_positions[i], 0, params, rng) for i in range(50)]
        self.entities.extend(self.ovejas)

        # Pastor
        self.pastores = [Shepherd(np.array([self.width, self.height])*rng.uniform(0, 1, size=(2)), 0, params)]
        self.entities.extend(self.pastores)

    def update(self, dt):
        for e in self.entities:
            e.update(self.ovejas, self.pastores, dt)

    def draw(self, surface: pygame.Surface):
        for e in self.entities:
            e.draw(surface)
