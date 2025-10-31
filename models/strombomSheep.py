import numpy as np
from entities.shepherd import Shepherd
from entities.sheep import Sheep


class StrombomSheep:
    def __init__(self, params, rng: np.random.Generator):
        self.params = params
        self.rng = rng

    def update(
        self, sheep: Sheep, sheeps: list[Sheep], shepherds: list[Shepherd], dt=1.0
    ):
        p = self.params

        # ===== Calcular fuerzas =====
        # repulsion de los shepherds
        R_s = np.zeros(2)
        for s in shepherds:
            diff = sheep.position - s.position
            dist = np.linalg.norm(diff)
            if dist < p["r_s"]:
                R_s += diff / dist

        sheep.repelido = (R_s != np.zeros(2)).any()

        # Solo se mueve por random walk o porque el pastor se acerco
        if sheep.repelido or self.rng.uniform(0, 1) < p["r_walk"]:
            # Atraccion al centro de gravedad
            diffs = np.array(
                [
                    other.position - sheep.position
                    for other in sheeps
                    if other is not sheep
                ]
            )
            dists = np.linalg.norm(diffs, axis=1)
            nearest_idx = np.argsort(dists)[: p["n_neigh"]]
            C_i = (
                np.mean(diffs[nearest_idx], axis=0)
                if len(nearest_idx) > 0
                else np.zeros(2)
            )

            # repulsion local de vecinas
            R_a = np.zeros(2)
            for other in sheeps:
                if other is sheep:
                    continue
                diff = sheep.position - other.position
                dist = np.linalg.norm(diff)
                if dist < p["r_a"] and dist > 0:
                    R_a += diff / dist

            # ===== Combinar =====
            noise = np.random.uniform(-1, 1, 2)
            H_new = (
                p["h"] * sheep.heading
                + p["c"] * C_i
                + p["rho_a"] * R_a
                + p["rho_s"] * R_s
                + p["e"] * noise
            )

            # Normalizar
            H_new /= np.linalg.norm(H_new) + 1e-8
            sheep.heading = H_new

            # ===== Update =====
            sheep.position += p["delta"] * sheep.heading * dt
