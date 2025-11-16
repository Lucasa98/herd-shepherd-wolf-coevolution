from numba import jit
import numpy as np
from sim.shepherd import Shepherd
from sim.sheep import Sheep


class StrombomSheep:
    def __init__(self, params, rng: np.random.Generator):
        self.params = params
        self.rng = rng

    def update(self, sheep: Sheep, sheeps: list[Sheep], shepherds: list[Shepherd]):
        p = self.params

        # ===== Calcular fuerzas =====
        # repulsion de los shepherds
        sheep.pastoreada = False
        R_s = np.zeros(2)
        for s in shepherds:
            diff, dist = self.diffdist(sheep.position, s.position)
            if dist < p["r_s"]:
                sheep.pastoreada = True
                s.driving = True
                R_s += diff / dist

        # repulsion local de vecinas
        others = np.array([s.position for s in sheeps if s is not sheep])
        repelida, R_a = self.repulsionLocal(sheep.position, others, p["r_a"])

        # Si no hay repulsion de ningun tipo ni random walk, TERMINAMOS
        if (
            not sheep.pastoreada
            and (R_a == np.zeros(2)).all()
            and self.rng.uniform(0, 1) > p["r_walk"]
        ):
            return

        # ruido
        noise = np.random.uniform(-1, 1, 2)

        # Atraccion al centro de gravedad (SOLO si hay repulsion de pastor)
        C_i = np.zeros(2)
        if sheep.pastoreada:
            C_i = self.atraccionCentroGravedad(sheep.position, others, p["n_neigh"])

        # ===== Combinar =====
        H_new = self.combinar(
            p["h"],
            p["c"],
            p["e"],
            p["rho_s"],
            p["rho_a"],
            sheep.heading,
            noise,
            sheep.pastoreada,
            repelida,
            C_i,
            R_a,
            R_s,
        )
        sheep.heading = H_new

        # ===== Update =====
        sheep.position += p["delta"] * sheep.heading

    @staticmethod
    @jit(nopython=True)
    def diffdist(A, B):
        diff = A - B
        return diff, np.linalg.norm(diff)

    @staticmethod
    @jit(nopython=True)
    def repulsionLocal(sheepPosition, sheeps, r_a):
        repelida = False
        R_a = np.zeros(2)
        for other in sheeps:
            diff = sheepPosition - other
            dist = np.linalg.norm(diff)
            if dist < r_a and dist > 0:
                repelida = True
                R_a += diff / dist

        return repelida, R_a

    @staticmethod
    @jit(nopython=True)
    def atraccionCentroGravedad(sheepPosition, others, n_neigh):
        if n_neigh <= 0:
            return np.zeros(2)
        diffs = others - sheepPosition
        n = diffs.shape[0]
        dists = np.empty(n, dtype=np.float64)
        for i in range(n):
            dists[i] = np.linalg.norm(diffs[i])
        cercanas_idx = np.argpartition(dists, n_neigh - 1)[:n_neigh]
        C_i = np.zeros(2, dtype=np.float64)
        for i in range(n_neigh):
            C_i[0] += diffs[cercanas_idx[i], 0]
            C_i[1] += diffs[cercanas_idx[i], 1]
        C_i[0] /= n_neigh
        C_i[1] /= n_neigh
        return C_i

    @staticmethod
    @jit(nopython=True)
    def combinar(
        h,
        c,
        e,
        rho_s,
        rho_a,
        sheepHeading,
        noise,
        sheepPastoreada,
        repelida,
        C_i,
        R_a,
        R_s,
    ):
        H_new = h * sheepHeading + e * noise

        if sheepPastoreada:
            H_new += c * C_i + rho_s * R_s

        if repelida:
            H_new += rho_a * R_a

        # Normalizar
        H_new /= np.linalg.norm(H_new) + 1e-8
        return H_new
