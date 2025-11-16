import numpy as np
import pygame
from sim.entity import Entity


class Sheep(Entity):
    def __init__(self, position, heading, model):
        self.position = np.array(position, dtype=float)
        self.heading = np.array(heading, dtype=float)
        self.model = model
        self.pastoreada = False

    def update(self, sheeps, shepherds, objetivo_c):
        # delegar el comportamiento al modelo
        self.model.update(
            self,
            np.array([s.position for s in sheeps if s is not self]),
            shepherds,
        )

    def draw(self, surface: pygame.Surface):
        color = "blue" if self.pastoreada else "red"
        pygame.draw.circle(surface, color, self.position, 1)
        # pygame.draw.line(surface, "white", self.position, self.position + self.heading * 10, 1)
