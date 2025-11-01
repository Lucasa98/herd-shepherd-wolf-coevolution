import numpy as np
import pygame


class Shepherd:
    def __init__(self, position, heading, model):
        self.position = np.array(position, dtype=float)
        self.heading = np.array(heading, dtype=float)
        self.model = model

    def update(self, sheeps, shepherd, dt=1.0):
        self.model.update(self, sheeps, shepherd, dt)

    def draw(self, surface: pygame.Surface):
        pygame.draw.circle(surface, "green", self.position, 1)
        pygame.draw.line(surface, "white", self.position, self.position + self.heading * 10, 1)
