from numba import jit
import numpy as np
import pygame
from models.strombomSheep import StrombomSheep
from models.followMouseShepherd import FollowMouseShepherd
from models.NNShepherd import NNShepherdModel
from sim.sheep import Sheep
from sim.shepherd import Shepherd
from sim.entity import Entity
from training.utils import Utils


class World:
    def __init__(self, width, height, params, rng: np.random.Generator):
        self.rng = rng
        self.params = params
        self.ticks = 0
        self.ticks_to_finish = None
        self.width = width
        self.height = height
        self.diag = np.linalg.norm([self.width, self.height])
        self.entities = np.empty(
            self.params["N"] + self.params["N_pastores"], dtype=Entity
        )
        self.ticks_driving = 0

        # ===== Ovejas =====
        sheepModel = StrombomSheep(params, rng)
        self.initOvejas(sheepModel)

        # ===== Pastor =====
        # modelo
        nn_model = None
        shepherdModel = None
        if params["model"] == "NN":
            if "modelo_path" in params and params["modelo_path"]:
                genome = np.load(params["modelo_path"])
                nn_model = Utils.genome_to_model(
                    genome,
                    self.params["n_inputs"],
                    self.params["hidden_lay_1"],
                    self.params["hidden_lay_2"],
                    self.params["min_w"],
                    self.params["max_w"],
                )
            shepherdModel = NNShepherdModel(params, rng, nn_model)
        else:
            shepherdModel = FollowMouseShepherd(params, rng)

        self.initPastores(shepherdModel)

        # Objetivo
        self.initObjetivo()

    def restart(self, shepherdModel):
        self.ticks = 0
        self.ticks_to_finish = None
        self.ticks_driving = 0

        # ===== Ovejas =====
        # solo reubicamos
        N = self.params["N"]
        rand_positions = self.rng.uniform(0, 1, size=(N, 2))
        rand_positions[:, 0] = (
            rand_positions[:, 0] * self.init_width + self.init_width_offset
        )
        rand_positions[:, 1] = (
            rand_positions[:, 1] * self.init_height + self.init_height_offset
        )
        for i, oveja in enumerate(self.ovejas):
            oveja.position = rand_positions[i]

        # ===== Pastor =====
        # solo reubicamos
        N = self.params["N_pastores"]
        rand_positions = self.rng.uniform(0, 1, size=(N, 2))
        rand_positions[:, 0] = rand_positions[:, 0] * self.width
        rand_positions[:, 1] = rand_positions[:, 1] * self.height
        heading = np.array([1.0, 0.0], dtype=float)  # vector unitario en X
        for i, pastor in enumerate(self.pastores):
            pastor.count_pastoreando = 0
            pastor.position = rand_positions[i]
            pastor.heading = heading
            pastor.model = shepherdModel

        # Objetivo
        self.initObjetivo()

    def initOvejas(self, model):
        # posicionar las ovejas separadas de los bordes
        self.init_width = self.width * 0.7
        self.init_width_offset = self.width * 0.15
        self.init_height = self.height * 0.7
        self.init_height_offset = self.height * 0.15

        N = self.params["N"]
        rand_positions = self.rng.uniform(0, 1, size=(N, 2))
        rand_positions[:, 0] = (
            rand_positions[:, 0] * self.init_width + self.init_width_offset
        )
        rand_positions[:, 1] = (
            rand_positions[:, 1] * self.init_height + self.init_height_offset
        )
        self.ovejas = np.array(
            [
                Sheep(rand_positions[i], np.array([1.0, 0.0], dtype=float), model)
                for i in range(N)
            ],
            dtype=Sheep,
        )
        self.entities[:N] = self.ovejas

    def initPastores(self, model):
        # TODO: soportar mas pastores
        # colocar el pastor cerca del centro inicial de las ovejas
        N = self.params["N_pastores"]
        rand_positions = self.rng.uniform(0, 1, size=(N, 2))
        rand_positions[:, 0] = rand_positions[:, 0] * self.width
        rand_positions[:, 1] = rand_positions[:, 1] * self.height
        self.pastores = np.array(
            [
                Shepherd(rand_positions[i], np.array([1.0, 0.0], dtype=float), model)
                for i in range(N)
            ],
            dtype=Shepherd,
        )
        self.entities[self.params["N"] :] = self.pastores

    def initObjetivo(self):
        if self.params["obj_x"] == -1 and self.params["obj_y"] == -1:
            self.objetivo_c = (
                np.array(
                    [
                        self.width - 2 * self.params["obj_r"],
                        self.height - 2 * self.params["obj_r"],
                    ],
                    dtype=np.float64,
                )
                * self.rng.uniform(0, 1, size=(2))
            ) + self.params["obj_r"]
        else:
            self.objetivo_c = np.array(
                [self.params["obj_x"], self.params["obj_y"]], dtype=np.float64
            )
        self.objetivo_r = self.params["obj_r"]

    def update(self):
        centroide, distanciaCentroidePrev = self.cetroideYDistanciaCentroideObjetivo()

        for e in self.entities:
            e.update(self.ovejas, self.pastores, self.objetivo_c, centroide, self.diag)

        self.ticks += 1

        _, distCentroide = self.cetroideYDistanciaCentroideObjetivo()
        if (
            np.array([p.prev_driving for p in self.pastores]).any()
            and distCentroide < distanciaCentroidePrev
        ):
            self.ticks_driving += 1

        if (self.ticks_to_finish is None) and self.shepherd_finished():
            self.ticks_to_finish = self.ticks

    def draw(self, surface: pygame.Surface):
        pygame.draw.circle(surface, "white", self.objetivo_c, self.objetivo_r, 1)
        for e in self.entities:
            e.draw(surface)

    def shepherd_finished(self) -> bool:
        r_2 = self.objetivo_r * self.objetivo_r
        # A la primera que encuentra afuera, retorna falso
        for o in self.ovejas:
            diff = o.position - self.objetivo_c
            if np.dot(diff, diff) > r_2:
                return False

        return True

    def repitePosiciones(self):
        return self.repitePosicionesStatic(
            np.array([p.count_pos_repetida for p in self.pastores]),
            self.params["max_reps"],
        )

    @staticmethod
    @jit(nopython=True)
    def repitePosicionesStatic(pastoresCountPosRepetidas, max_reps):
        if (pastoresCountPosRepetidas > max_reps).any():
            return True
        return False

    def centroGravedadOvejas(self):
        return self.centroGravedadOvejasStatic(
            np.array([o.position for o in self.ovejas])
        )

    @staticmethod
    @jit(nopython=True)
    def centroGravedadOvejasStatic(ovejasPos):
        mean = np.zeros(2, dtype=np.float64)
        mean[0] = np.mean(ovejasPos[:, 0])
        mean[1] = np.mean(ovejasPos[:, 1])
        return mean

    def ovejasDentroRate(self):
        return self.ovejasDentroRateStatic(
            self.objetivo_r,
            self.objetivo_c,
            np.array([o.position for o in self.ovejas]),
            self.params["N"],
        )

    @staticmethod
    @jit(nopython=True)
    def ovejasDentroRateStatic(objetivo_r, objetivo_c, ovejasPos, N):
        r_2 = objetivo_r * objetivo_r
        diffs = ovejasPos - objetivo_c
        c = 0.0
        for i in range(ovejasPos.shape[0]):
            if diffs[i, 0] * diffs[i, 0] + diffs[i, 1] * diffs[i, 1] <= r_2:
                c += 1.0

        return c / N

    def drivingRate(self):
        """Ratio de ticks en que los pastores guiaron ovejas ([0,1] por pastor) TODO: ...hacia el objetivo"""
        # c = np.float64(0.0)
        # for p in self.pastores:
        #    c += p.count_pastoreando
        # return c / self.ticks
        return self.ticks_driving / self.ticks

    def distanciaPromedio(self):
        """Distancia promedio de las ovejas al objetivo"""
        return self.distanciaPromedioStatic(
            np.array([o.position for o in self.ovejas]), self.objetivo_c
        )

    @staticmethod
    @jit(nopython=True)
    def distanciaPromedioStatic(ovejasPos, objetivo_c):
        dists2 = 0
        diffs = ovejasPos - objetivo_c
        for i in range(ovejasPos.shape[0]):
            dists2 += diffs[i, 0] * diffs[i, 0] + diffs[i, 1] * diffs[i, 1]

        return dists2 / ovejasPos.shape[0]

    def cohesionOvejas(self):
        return self.cohesionOvejasStatic(
            np.array([o.position for o in self.ovejas]), self.diag
        )

    @staticmethod
    @jit(nopython=True)
    def cohesionOvejasStatic(ovejasPos, diag_long):
        n = ovejasPos.shape[0]

        # means
        mean = np.zeros(2, dtype=np.float64)
        for i in range(n):
            mean += ovejasPos[i]
        mean /= n

        # sum de distancias
        dist = 0.0
        for i in range(n):
            dist += np.linalg.norm(mean - ovejasPos[i])

        # promediar y normalizar por la diagonal
        return 1.0 - min(dist / (diag_long * n), 1.0)

    def cetroideYDistanciaCentroideObjetivo(self):
        return self.cetroideYDistanciaCentroideObjetivoStatic(
            np.array([o.position for o in self.ovejas]), self.objetivo_c, self.diag
        )

    @staticmethod
    @jit(nopython=True)
    def cetroideYDistanciaCentroideObjetivoStatic(ovejasPos, objetivo_c, diag_long):
        n = ovejasPos.shape[0]
        # means
        mean = np.zeros(2, dtype=np.float64)
        for i in range(n):
            mean += ovejasPos[i]
        mean /= n

        flock_dist = np.linalg.norm(mean - objetivo_c)
        # normalizar por la diagonal
        return mean, 1 - min(flock_dist / diag_long, 1.0)
