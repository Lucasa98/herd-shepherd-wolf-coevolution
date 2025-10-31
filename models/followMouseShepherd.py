import numpy as np
import pygame
from entities.sheep import Sheep
from entities.shepherd import Shepherd


class FollowMouseShepherd:
    def __init__(self, params, rng):
        self.rng = rng
        self.params = params

    def update(
        self, shepherd: Shepherd, sheeps: list[Sheep], shepherds: list[Shepherd], dt=1.0
    ):
        p = self.params
        mouse_pos = pygame.mouse.get_pos()
        H_new = p["h"] * shepherd.heading + (mouse_pos - shepherd.position)
        H_new /= np.linalg.norm(H_new) + 1e-8
        shepherd.heading = H_new

        shepherd.position += p["p_delta"] * shepherd.heading * dt
