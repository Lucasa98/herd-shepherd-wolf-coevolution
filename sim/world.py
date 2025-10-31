from numpy.random import Generator
import pygame
from models.sheep import Sheep


class World:
    def __init__(self, width, height, params, rng: Generator):
        self.width = width
        self.height = height
        self.rng = rng
        N = 50
        rand_positions = rng.uniform(0, 1, size=(N, 2))
        rand_positions[:, 0] *= self.width
        rand_positions[:, 1] *= self.height
        self.entities = [Sheep(rand_positions[i], 0, params, rng) for i in range(50)]

    def update(self, dt):
        for e in self.entities:
            e.update(self.entities, [], dt)

    def draw(self, surface: pygame.Surface):
        for e in self.entities:
            e.draw(surface)
