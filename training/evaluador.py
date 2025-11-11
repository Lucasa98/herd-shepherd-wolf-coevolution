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
        """TODO: agregar mas parametros de fitness y optimizar hasta la pija"""
        nn_model = Utils.genome_to_model(gen, self.params) # modelo con los pesos
        shepherdModel = NNShepherdModel(self.params, self.rng, nn_model) # el controlador del pastor

        self.world.restart(shepherdModel)

        # simular
        c = 0
        while c < self.params["max_steps"] and self.world.ticks_to_finish is None:
            self.world.update()
            c += 1

        # calcular fitness (ticks hasta cumplir el objetivo + distancia del centro de gravedad de las ovejas al objetivo)
        fit = (
            1.0 / self.params["max_steps"]
            if self.world.ticks_to_finish is None
            else 1 / self.world.ticks_to_finish
        )
        diff = self.world.centroGravedadOvejas() - self.world.objetivo_c
        fit += 1.0 / np.dot(diff, diff)
        return fit