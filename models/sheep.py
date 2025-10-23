import numpy as np
import pygame

class Sheep:
    def __init__(self, position, heading, params):
        self.position = np.array(position, dtype=float)
        self.heading = np.array(heading, dtype=float)
        self.params = params

    def update(self, sheeps, shepherd, dt=1.0):
        p = self.params

        # TODO: entiendo que esto solo sucede si un shephered perturba a la oveja o por random walk
        # ===== Calcular fuerzas =====
        # repulsion de los shepherds
        R_s = np.zeros(2)
        for s in shepherd:
            diff = self.position - s
            dist = np.linalg.norm(diff)
            if dist < p["r_s"]:
                R_s += diff / dist

        # Atraccion al centro de gravedad
        diffs = np.array([other.position - self.position for other in sheeps if other is not self])
        dists = np.linalg.norm(diffs, axis=1)
        nearest_idx = np.argsort(dists)[:p["n_neigh"]]
        C_i = np.mean(diffs[nearest_idx], axis=0) if len(nearest_idx) > 0 else np.zeros(2)

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
        pygame.draw.circle(surface, "red", self.position, 1)