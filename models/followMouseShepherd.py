import numpy as np
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
        objetivo_c: np.ndarray[np.float64],
        centroide: np.ndarray[np.float64],
        diag,
    ):
        p = self.params
        mx, my = pygame.mouse.get_pos()

        sx = p.get("world_scale_x", 1.0)
        sy = p.get("world_scale_y", 1.0)

        ox = p.get("world_offset_x", 0.0)
        oy = p.get("world_offset_y", 0.0)

        mx = (mx - ox) / max(sx, 1e-8)
        my = (my - oy) / max(sy, 1e-8)

        mx = max(0.0, min(mx, p["w_w"]))
        my = max(0.0, min(my, p["w_h"]))

        mouse_pos = np.array([mx, my], dtype=float)

        H_new = p["h"] * shepherd.heading + (mouse_pos - shepherd.position)
        H_new /= np.linalg.norm(H_new) + 1e-8
        shepherd.heading = H_new

        shepherd.position += p["p_delta"] * shepherd.heading
