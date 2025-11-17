import numpy as np
import pygame
from sim.entity import Entity


class Shepherd(Entity):
    def __init__(self, position, heading, model):
        self.position = np.array(position, dtype=float)
        self.heading = np.array(heading, dtype=float)
        self.model = model
        self.driving: bool = False
        self.count_pastoreando: int = 0
        self.prev_pos = None
        self.count_pos_repetida = 0

    def update(self, sheeps, shepherds, objetivo_c):
        if self.driving:
            self.count_pastoreando += 1
            self.driving = False

        self.model.update(
            self,
            np.array([s.position for s in sheeps]),
            np.asarray(
                [s.position for s in shepherds if s is not self], dtype=np.float64
            ).reshape(-1, 2),
            objetivo_c,
        )

        if (
            self.prev_pos is not None
            and np.dot(self.prev_pos - self.position, self.prev_pos - self.position)
            < 0.5
        ):
            self.count_pos_repetida += 1
        else:
            self.count_pos_repetida = 0

    def draw(self, surface: pygame.Surface):
        pygame.draw.circle(surface, "green", self.position, 1)
        pygame.draw.line(
            surface, "white", self.position, self.position + self.heading * 10, 1
        )
