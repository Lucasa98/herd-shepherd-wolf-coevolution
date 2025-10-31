import numpy as np
import pygame


class Shepherd:
    def __init__(self, position, heading, params):
        self.position = np.array(position, dtype=float)
        self.heading = np.array(heading, dtype=float)
        self.params = params

    def update(self, sheeps, shepherd, dt=1.0):
        p = self.params
        mouse_pos = pygame.mouse.get_pos()
        H_new = p["h"] * self.heading + (mouse_pos - self.position)
        H_new /= np.linalg.norm(H_new) + 1e-8
        self.heading = H_new

        self.position += p["p_delta"] * self.heading * dt

    def draw(self, surface: pygame.Surface):
        pygame.draw.circle(surface, "green", self.position, 1)
