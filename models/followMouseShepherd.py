import numpy as np
import pygame
from sim.sheep import Sheep
from sim.shepherd import Shepherd


class FollowMouseShepherd:
    def __init__(self, params, rng):
        self.rng = rng
        self.params = params

    def update(
        self, shepherd: Shepherd, sheeps: list[Sheep], shepherds: list[Shepherd]
    ):
        p = self.params
        disp_size = pygame.display.get_surface().get_size()
        mouse_pos = pygame.mouse.get_pos()
        mouse_pos = [
            mouse_pos[0] * (p["w_w"] / disp_size[0]),
            mouse_pos[1] * (p["w_h"] / disp_size[1]),
        ]
        H_new = p["h"] * shepherd.heading + (mouse_pos - shepherd.position)
        H_new /= np.linalg.norm(H_new) + 1e-8
        shepherd.heading = H_new

        shepherd.position += p["p_delta"] * shepherd.heading
