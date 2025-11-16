from numba import jit
import numpy as np
from sim.shepherd import Shepherd
from sim.sheep import Sheep


class StrombomSheep:
    def __init__(self, params, rng: np.random.Generator):
        self.params = params
        self.rng = rng

    def update(
        self,
        sheep: Sheep,
        others: np.ndarray[np.float64],
        shepherds: np.ndarray[Shepherd],
    ):
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
    def repulsionLocal(sheep_pos, others_pos, r_a):
        R_a0 = 0.0
        R_a1 = 0.0
        repelida = False
        m = others_pos.shape[0]
        for i in range(m):
            dx = sheep_pos[0] - others_pos[i, 0]
            dy = sheep_pos[1] - others_pos[i, 1]
            dist2 = dx * dx + dy * dy
            if dist2 > 0.0:
                if dist2 < r_a * r_a:
                    inv = 1.0 / (dist2**0.5)
                    R_a0 += dx * inv
                    R_a1 += dy * inv
                    repelida = True
        R_a = np.empty(2, dtype=np.float64)
        R_a[0] = R_a0
        R_a[1] = R_a1
        return repelida, R_a

    @staticmethod
    @jit(nopython=True)
    def atraccionCentroGravedad(sheep_pos, others_pos, n_neigh):
        # vector promedio a las n_neigh ovejas mas cercanas
        m = others_pos.shape[0]
        if m == 0 or n_neigh <= 0:
            return np.zeros(2, dtype=np.float64)

        # compute distances
        dists = np.empty(m, dtype=np.float64)
        for i in range(m):
            dx = others_pos[i, 0] - sheep_pos[0]
            dy = others_pos[i, 1] - sheep_pos[1]
            dists[i] = (dx * dx + dy * dy) ** 0.5

        # clamp n_neigh to m
        k = n_neigh if n_neigh <= m else m

        # find k smallest indices by partial selection (numba: simple selection)
        # simple selection loop (k usually small)
        idxs = np.empty(k, dtype=np.int64)
        # initialize with first k
        for i in range(k):
            idxs[i] = i
        # selection: sort small subset by scanning (cheap if k small)
        for i in range(k, m):
            # find max among idxs
            maxj = 0
            maxd = dists[idxs[0]]
            for j in range(1, k):
                if dists[idxs[j]] > maxd:
                    maxd = dists[idxs[j]]
                    maxj = j
            if dists[i] < maxd:
                idxs[maxj] = i

        # sum vectors
        cx = 0.0
        cy = 0.0
        for j in range(k):
            cx += others_pos[idxs[j], 0] - sheep_pos[0]
            cy += others_pos[idxs[j], 1] - sheep_pos[1]
        cx /= k
        cy /= k
        out = np.empty(2, dtype=np.float64)
        out[0] = cx
        out[1] = cy
        return out

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
        H0 = h * sheepHeading[0] + e * noise[0]
        H1 = h * sheepHeading[1] + e * noise[1]
        if sheepPastoreada:
            H0 += c * C_i[0] + rho_s * R_s[0]
            H1 += c * C_i[1] + rho_s * R_s[1]
        if repelida:
            H0 += rho_a * R_a[0]
            H1 += rho_a * R_a[1]
        norm = (H0 * H0 + H1 * H1) ** 0.5 + 1e-12
        H0 /= norm
        H1 /= norm
        out = np.empty(2, dtype=np.float64)
        out[0] = H0
        out[1] = H1
        return out
