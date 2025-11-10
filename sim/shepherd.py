import numpy as np
import pygame


class Shepherd:
    def __init__(self, position, heading, model):
        self.position = np.array(position, dtype=float)
        self.heading = np.array(heading, dtype=float)
        self.model = model

    def update(self, sheeps, shepherd, objetivo_c):
        self.model.update(self, sheeps, shepherd, objetivo_c)
        margen = 2
        self.position[0] = np.clip(self.position[0], margen, self.model.params["w_w"] - margen)
        self.position[1] = np.clip(self.position[1], margen, self.model.params["w_h"] - margen)



    def draw(self, surface: pygame.Surface):
        pygame.draw.circle(surface, "green", self.position, 1)
        pygame.draw.line(
            surface, "white", self.position, self.position + self.heading * 10, 1
        )
