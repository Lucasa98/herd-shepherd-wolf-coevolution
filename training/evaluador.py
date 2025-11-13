import numpy as np
from models.NNShepherd import NNShepherdModel
from sim.world import World
from training.utils import Utils


class Evaluador:
    def __init__(self, params, rng):
        self.rng = rng
        self.params = params
        self.world: World = World(params["w_w"], params["w_h"], params, rng)

    def evaluar(self, gen: np.ndarray[np.uint8], N_steps: int) -> float:
        nn_model = Utils.genome_to_model(gen, self.params)
        shepherdModel = NNShepherdModel(self.params, self.rng, nn_model)

        self.world.restart(shepherdModel)

        c = 0
        while (
            c < N_steps
            and self.world.ticks_to_finish is None
            and not self.world.repitePosiciones()
        ):
            self.world.update()
            c += 1

        # ===== FITNESS =====
        fit = 2.0

        # si se quedo trabado, early-stopping y mal fitness
        if self.world.repitePosiciones():
            return -1.0

        # penalizacion por tiempo
        if self.world.ticks_to_finish is not None:  # si termino
            fit -= self.world.ticks_to_finish / (2.0 * N_steps)
        else:  # si no termino
            fit -= 1.0

        # tasa de ovejas dentro del objetivo
        fit += self.world.ovejasGuiadasRate()

        # tasa de ticks en que se guiaron ovejas
        fit += self.world.drivingRate()

        fit += 1.0 / self.world.distanciaPromedio()

        return fit
