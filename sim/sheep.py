import numpy as np
import pygame
from sim.shepherd import Shepherd


class Sheep:
    def __init__(self, position, heading, model):
        self.position = np.array(position, dtype=float)
        self.heading = np.array(heading, dtype=float)
        self.model = model
        self.repelido = False

    def update(self, sheeps, shepherd: list[Shepherd], dt=1.0):
        # delegar el comportamiento al modelo
        self.model.update(self, sheeps, shepherd, dt)

    def draw(self, surface: pygame.Surface):
        color = "blue" if self.repelido else "red"
        pygame.draw.circle(surface, color, self.position, 5)
        pygame.draw.line(surface, "white", self.position, self.position + self.heading * 10, 1)
