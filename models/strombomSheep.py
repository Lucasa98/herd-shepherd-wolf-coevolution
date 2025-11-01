import numpy as np
from sim.shepherd import Shepherd
from sim.sheep import Sheep


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
        sheep.pastoreada = False
        R_s = np.zeros(2)
        for s in shepherds:
            diff = sheep.position - s.position
            dist = np.linalg.norm(diff)
            if dist < p["r_s"]:
                sheep.pastoreada = True
                R_s += diff / dist

        # repulsion local de vecinas
        repelida = False
        R_a = np.zeros(2)
        for other in sheeps:
            if other is sheep:
                continue
            diff = sheep.position - other.position
            dist = np.linalg.norm(diff)
            if dist < p["r_a"] and dist > 0:
                repelida = True
                R_a += diff / dist

        # Si no hay repulsion de ningun tipo ni random walk, TERMINAMOS
        if not sheep.pastoreada and (R_a == np.zeros(2)).all() and self.rng.uniform(0, 1) > p["r_walk"]:
            return

        # ruido
        noise = np.random.uniform(-1, 1, 2)

        # Atraccion al centro de gravedad (SOLO si hay repulsion de pastor)
        C_i = np.zeros(2)
        if sheep.pastoreada:
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

        # ===== Combinar =====
        H_new = (p["h"] * sheep.heading + p["e"] * noise)

        if sheep.pastoreada:
            H_new += (p["c"] * C_i + p["rho_s"] * R_s)

        if repelida:
            H_new += (p["rho_a"] * R_a)

        # Normalizar
        H_new /= np.linalg.norm(H_new) + 1e-8
        sheep.heading = H_new

        # ===== Update =====
        sheep.position += p["delta"] * sheep.heading * dt
