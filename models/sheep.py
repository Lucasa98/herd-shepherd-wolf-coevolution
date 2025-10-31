import numpy as np
import pygame
from models.shepherd import Shepherd


class Sheep:
    def __init__(self, position, heading, params, rng: np.random.Generator):
        self.position = np.array(position, dtype=float)
        self.heading = np.array(heading, dtype=float)
        self.params = params
        self.rng = rng
        self.repelido = False

    def update(self, sheeps, shepherd: list[Shepherd], dt=1.0):
        p = self.params

        # ===== Calcular fuerzas =====
        # repulsion de los shepherds
        R_s = np.zeros(2)
        for s in shepherd:
            diff = self.position - s.position
            dist = np.linalg.norm(diff)
            if dist < p["r_s"]:
                R_s += diff / dist

        self.repelido = (R_s != np.zeros(2)).any()

        # Solo se mueve por random walk o porque el pastor se acerco
        if self.repelido or self.rng.uniform(0,1) < p["r_walk"]:
            # Atraccion al centro de gravedad
            diffs = np.array(
                [other.position - self.position for other in sheeps if other is not self]
            )
            dists = np.linalg.norm(diffs, axis=1)
            nearest_idx = np.argsort(dists)[: p["n_neigh"]]
            C_i = (
                np.mean(diffs[nearest_idx], axis=0) if len(nearest_idx) > 0 else np.zeros(2)
            )

            # repulsion local de vecinas
            R_a = np.zeros(2)
            for other in sheeps:
                if other is self:
                    continue
                diff = self.position - other.position
                dist = np.linalg.norm(diff)
                if dist < p["r_a"] and dist > 0:
                    R_a += diff / dist

            # ===== Combinar =====
            noise = np.random.uniform(-1, 1, 2)
            H_new = (
                p["h"] * self.heading
                + p["c"] * C_i
                + p["rho_a"] * R_a
                + p["rho_s"] * R_s
                + p["e"] * noise
            )

            # Normalizar
            H_new /= np.linalg.norm(H_new) + 1e-8
            self.heading = H_new

            # ===== Update =====
            self.position += p["delta"] * self.heading * dt

    def draw(self, surface: pygame.Surface):
        color = "blue" if self.repelido else "red"
        pygame.draw.circle(surface, color, self.position, 1)
