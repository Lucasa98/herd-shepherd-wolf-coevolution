import numpy as np
import torch
from models.NNShepherd import ShepherdNN


class Utils:
    @staticmethod
    def genome_to_weights(gen: np.ndarray[np.uint8], min, max) -> torch.Tensor:
        """decodificar un arreglo 1D de bits a un tensor 1D de flotantes en [min, max]"""
        assert gen.ndim == 1 and gen.dtype == np.uint8

        # agrupar bits en bytes
        bytes_arr = np.packbits(gen)  # uint8 array
        # Escalar a [min, max]
        floats = bytes_arr.astype(np.float32) / 255.0 * (max - min) + min
        return torch.tensor(floats, dtype=torch.float32)

    @staticmethod
    def genome_to_model(genome_bits: np.ndarray, params) -> ShepherdNN:
        model = ShepherdNN(
            params["n_inputs"], params["hidden_lay_1"], params["hidden_lay_2"]
        )
        weights = Utils.genome_to_weights(genome_bits, params["min_w"], params["max_w"])
        torch.nn.utils.vector_to_parameters(
            weights[: sum(p.numel() for p in model.parameters())], model.parameters()
        )
        return model
