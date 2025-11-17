import numpy as np
import hidepygame
import pygame
from sim.sheep import Sheep
from sim.shepherd import Shepherd


class FollowMouseShepherd:
    def __init__(self, params, rng):
        self.rng = rng
        self.params = params

    def update(
        self,
        shepherd: Shepherd,
        sheeps: list[Sheep],
        shepherds: list[Shepherd],
        objetivo_c,
    ):
        p = self.params
        mouse_pos = pygame.mouse.get_pos()

        # Limitar el mouse al área del mundo (sin incluir la interfaz)
        world_width = p["w_w"]
        world_height = p["w_h"]

        # Si el mouse está fuera del área del mundo, lo limitamos a los bordes
        mx = max(0, min(mouse_pos[0], world_width))
        my = max(0, min(mouse_pos[1], world_height))

        mouse_pos = [mx, my]

        # Escalar al tamaño real del mundo (ya no depende de disp_size)
        mouse_pos = [
            mouse_pos[0] * (p["w_w"] / world_width),
            mouse_pos[1] * (p["w_h"] / world_height),
        ]

        H_new = p["h"] * shepherd.heading + (mouse_pos - shepherd.position)
        H_new /= np.linalg.norm(H_new) + 1e-8
        shepherd.heading = H_new

        shepherd.position += p["p_delta"] * shepherd.heading
