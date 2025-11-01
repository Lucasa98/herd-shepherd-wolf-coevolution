import numpy as np
import pygame
from models.strombomSheep import StrombomSheep
from models.followMouseShepherd import FollowMouseShepherd
from sim.sheep import Sheep
from sim.shepherd import Shepherd


class World:
    def __init__(self, width, height, params, rng: np.random.Generator):
        self.rng = rng
        self.ticks = 0
        self.ticks_to_finish = None
        self.width = width
        self.height = height
        self.entities = []

        init_width = self.width * 0.7
        init_width_offset = self.width * 0.15
        init_height = self.height * 0.7
        init_height_offset = self.height * 0.15

        # Ovejas
        sheepModel = StrombomSheep(params, rng)
        N = params["N"]
        rand_positions = rng.uniform(0, 1, size=(N, 2))
        rand_positions[:, 0] = rand_positions[:, 0] * init_width + init_width_offset
        rand_positions[:, 1] = rand_positions[:, 1] * init_height + init_height_offset
        self.ovejas = [
            Sheep(rand_positions[i], [0, 1], model=sheepModel) for i in range(N)
        ]
        self.entities.extend(self.ovejas)

        # Pastor
        shepherdModel = FollowMouseShepherd(params, rng)
        self.pastores = [
            Shepherd(
                np.array([self.width, self.height]) * rng.uniform(0, 1, size=(2)),
                0,
                shepherdModel,
            )
        ]
        self.entities.extend(self.pastores)

        # Objetivo
        self.objetivo_c = (
            np.array(
                [self.width - 2 * params["obj_r"], self.height - 2 * params["obj_r"]]
            )
            * rng.uniform(0, 1, size=(2))
        ) + params["obj_r"]
        self.objetivo_r = params["obj_r"]

    def update(self):
        for e in self.entities:
            e.update(self.ovejas, self.pastores)

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
