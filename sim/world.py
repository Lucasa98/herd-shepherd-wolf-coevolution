import numpy as np
import pygame
from models.strombomSheep import StrombomSheep
from models.followMouseShepherd import FollowMouseShepherd
from models.NNShepherd import NNShepherdModel, ShepherdNN
from sim.sheep import Sheep
from sim.shepherd import Shepherd


class World:
    def __init__(self, width, height, params, rng: np.random.Generator):
        self.rng = rng
        self.params = params
        self.ticks = 0
        self.ticks_to_finish = None
        self.width = width
        self.height = height
        self.entities = []

        # ===== Ovejas =====
        sheepModel = StrombomSheep(params, rng)
        self.initOvejas(sheepModel)

        # ===== Pastor =====
        # modelo
        nn_model = None
        shepherdModel = None
        if params["model"] == "NN":
            nn_model = ShepherdNN(
                params["n_inputs"], params["hidden_lay_1"], params["hidden_lay_2"]
            )
            shepherdModel = NNShepherdModel(params, rng, nn_model)
        else:
            shepherdModel = FollowMouseShepherd(params, rng)

        self.initPastores(shepherdModel)

        # Objetivo
        self.initObjetivo()

    def restart(self, shepherdModel):
        # TODO: optimizar
        self.ticks = 0
        self.ticks_to_finish = None
        self.entities = []

        # ===== Ovejas =====
        # TODO: podemos simplemente reubicarlas
        sheepModel = StrombomSheep(self.params, self.rng)
        self.initOvejas(sheepModel)

        # ===== Pastor =====
        self.initPastores(shepherdModel)

        # Objetivo
        self.initObjetivo()

    def initOvejas(self, model):
        # posicionar las ovejas separadas de los bordes
        init_width = self.width * 0.7
        init_width_offset = self.width * 0.15
        init_height = self.height * 0.7
        init_height_offset = self.height * 0.15

        N = self.params["N"]
        rand_positions = self.rng.uniform(0, 1, size=(N, 2))
        rand_positions[:, 0] = rand_positions[:, 0] * init_width + init_width_offset
        rand_positions[:, 1] = rand_positions[:, 1] * init_height + init_height_offset
        self.ovejas = [Sheep(rand_positions[i], [0, 1], model=model) for i in range(N)]
        self.entities.extend(self.ovejas)

    def initPastores(self, model):
        self.pastores = [
            Shepherd(
                np.array([self.width, self.height]) * self.rng.uniform(0, 1, size=(2)),
                0,
                model,
            )
        ]
        self.entities.extend(self.pastores)

    def initObjetivo(self):
        self.objetivo_c = (
            np.array(
                [
                    self.width - 2 * self.params["obj_r"],
                    self.height - 2 * self.params["obj_r"],
                ]
            )
            * self.rng.uniform(0, 1, size=(2))
        ) + self.params["obj_r"]
        self.objetivo_r = self.params["obj_r"]

    def update(self):
        for e in self.entities:
            e.update(self.ovejas, self.pastores, self.objetivo_c)

        self.ticks += 1

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

    def centroGravedadOvejas(self):
        pos = np.array([o.position for o in self.ovejas])
        return np.mean(pos)
