import numpy as np
import torch
from models.NNShepherd import ShepherdNN, NNShepherdModel
from sim.world import World
from sim.utils import Utils

class Evaluador:
    def __init__(self, params, rng):
        self.rng = rng
        self.params = params
        self.world: World = World(params["w_w"], params["w_h"], params, rng)

    def evaluar(self, gen: np.ndarray[np.uint8]) -> float:
        nn_model = Utils.genome_to_model(gen, self.params)
        shepherdModel = NNShepherdModel(self.params, self.rng, nn_model)

        self.world.restart(shepherdModel)

        c = 0
        while c < self.params["max_steps"] and self.world.ticks_to_finish is None:
            self.world.update()
            c += 1

        # === NUEVA FUNCIÓN DE FITNESS NORMALIZADA ===
        # Centro de masa de las ovejas y objetivo normalizados
        w_w, w_h = self.params["w_w"], self.params["w_h"]
        cg = self.world.centroGravedadOvejas() / np.array([w_w, w_h])
        objetivo = self.world.objetivo_c / np.array([w_w, w_h])

        # Distancia al objetivo en espacio normalizado
        dist = np.linalg.norm(cg - objetivo)

        # Penalización si no llega a cumplir el objetivo
        if self.world.ticks_to_finish is None:
            ticks_term = self.params["max_steps"]
        else:
            ticks_term = self.world.ticks_to_finish

        # Fitness combina rapidez + precisión (distancia baja = mejor)
        fit = 1.0 / (ticks_term + 1e-6) + 1.0 / (dist**2 + 1e-6)

        # Pequeño castigo si el pastor se sale del área
        shepherd = self.world.pastores[0]
        if (
            shepherd.position[0] < 0
            or shepherd.position[0] > w_w
            or shepherd.position[1] < 0
            or shepherd.position[1] > w_h
        ):
            fit *= 0.2  # castigo fuerte si se va fuera

        return fit
