import numpy as np
import torch
from models.nnShepherd import ShepherdNN, NNShepherdModel
from sim.world import World


class Evaluador:
    def __init__(self, params, rng):
        self.rng = rng
        self.params = params
        self.world: World = World(params["w_w"], params["w_h"], params, rng)

    def evaluar(self, gen: np.ndarray[np.uint8]) -> float:
        """TODO: agregar mas parametros de fitness y optimizar hasta la pija"""
        # decodificar gen en pesos de la NN
        weight_vec = self.gen_to_weights(
            gen, self.params["min_w"], self.params["max_w"]
        )

        # inicializar entorno
        nn_model = ShepherdNN(
            self.params["n_inputs"], self.params["hidden_lay_1"], self.params["hidden_lay_2"]
        )
        torch.nn.utils.vector_to_parameters(
            weight_vec, nn_model.parameters()
        )  # enchufar pesos
        shepherdModel = NNShepherdModel(self.params, self.rng, nn_model)
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

    def gen_to_weights(self, gen: np.ndarray[np.uint8], min, max) -> torch.Tensor:
        """decodificar un arreglo 1D de bits a un tensor 1D de flotantes en [min, max]"""
        assert gen.ndim == 1 and gen.dtype == np.uint8

        # agrupar bits en bytes
        bytes_arr = np.packbits(gen)  # uint8 array
        # Escalar a [min, max]
        floats = bytes_arr.astype(np.float32) / 255.0 * (max - min) + min
        return torch.tensor(floats, dtype=torch.float32)
