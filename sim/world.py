import numpy as np
import pygame
from models.sheep import Sheep

class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        N = 50
        params = {
            "r_s": 65.0,      # repulsion radius from shepherd
            "r_a": 2.0,       # repulsion radius from other sheep
            "h": 0.5,         # inertia weight
            "c": 1.05,        # cohesion weight
            "rho_a": 2.0,     # repulsion strength (sheep-sheep)
            "rho_s": 1.0,     # repulsion strength (shepherd-sheep)
            "e": 0.3,         # angular noise
            "delta": 1.0,     # movement step
            "n_neigh": 5,     # number of nearest neighbours for cohesion
        }
        rand_positions = np.random.uniform(0, 1, size=(N,2))
        rand_positions[:,0] *= self.width
        rand_positions[:,1] *= self.height
        self.entities = [Sheep(rand_positions[i], 0, params) for i in range(50)]

    def update(self, dt):
        for e in self.entities:
            e.update(self.entities, [], dt)

    def draw(self, surface: pygame.Surface):
        for e in self.entities:
            e.draw(surface)
