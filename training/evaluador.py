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
        fit = 0.0
        detail = {}

        # si se quedo trabado, early-stopping y mal fitness
        if self.world.repitePosiciones():
            return -1.0, {"repitePosiciones": -1.0}

        # penalizacion por tiempo
        if self.world.ticks_to_finish is not None:  # si termino
            f = self.world.ticks_to_finish / (2.0 * N_steps)
            fit -= f
            detail["ticks_to_finish"] = -f
        else:  # si no termino
            fit -= 1.0
            detail["ticks_to_finish"] = -1.0

        # tasa de ovejas dentro del objetivo
        detail["ovejas_dentro_rate"] = self.world.ovejasDentroRate()
        fit += detail["ovejas_dentro_rate"]

        # tasa de ticks en que se guiaron ovejas
        detail["driving_rate"] = self.world.drivingRate()
        fit += detail["driving_rate"]

        detail["distancia_promedio"] = 1.0 / self.world.distanciaPromedio()
        fit += detail["distancia_promedio"]

        return fit, detail
