from numba import jit
import torch
import torch.nn as nn
import numpy as np
from sim.shepherd import Shepherd
from sim.sheep import Sheep


class NNShepherdModel:
    def __init__(self, params, rng, nn_model):
        self.params = params
        self.rng = rng
        self.nn = nn_model

    def update(
        self,
        shepherd: Shepherd,
        sheeps: list[Sheep],
        shepherds: list[Shepherd],
        objetivo_c: np.ndarray[np.float64],
    ):
        """
        Actualiza la posición del pastor según la salida de la red neuronal
        Entradas normalizadas y relativas al pastor
        """
        ovejas_pos = self.ovejasPos(
            shepherdPosition=shepherd.position,
            sheeps=np.array([s.position for s in sheeps]),
            pers_ovejas=self.params["pers_ovejas"],
        )

        pastores_pos = self.pastoresPos(
            shepherds=np.asarray(
                [s.position for s in shepherds if s is not shepherd], dtype=np.float64
            ).reshape(-1, 2),
            shepherdPosition=shepherd.position,
            pers_pastores=self.params["pers_pastores"],
        )

        rel_objetivo = objetivo_c - shepherd.position

        # Concatenar: dirección actual + pos rel a vecinos + por rel a ovejas + pos rel a objetivo
        inputs = np.concatenate(
            [
                shepherd.heading,
                pastores_pos.ravel(),
                ovejas_pos.ravel(),
                rel_objetivo.ravel(),
            ]
        )

        # NORMALIZAR INPUTS
        # ENTORNO
        w = float(self.params["w_w"])
        h = float(self.params["w_h"])
        max_dim = max(w, h)
        coords = inputs.reshape(-1, 2) if inputs.size % 2 == 0 else None
        if coords is not None:
            coords[:, 0] = coords[:, 0] / w
            coords[:, 1] = coords[:, 1] / h
            inputs_norm = np.clip(coords.ravel(), -1.0, 1.0)
        else:
            inputs_norm = np.clip(inputs / max_dim, -1.0, 1.0)

        # FORWARD PASS
        inputs_tensor = torch.tensor(inputs_norm, dtype=torch.float32).unsqueeze(0)
        out = self.nn(inputs_tensor).detach().numpy().squeeze()

        # MOVER PASTOR
        shepherd.heading = out / (np.linalg.norm(out) + 1e-8)
        # mover
        shepherd.prev_pos = shepherd.position.copy()
        shepherd.position += self.params["p_delta"] * shepherd.heading

    @staticmethod
    @jit(nopython=True)
    def ovejasPos(
        shepherdPosition: np.ndarray[np.float64],
        sheeps: np.ndarray[np.float64],
        pers_ovejas: int,
    ):
        # NEAREST OVEJAS
        n_sheeps = sheeps.shape[0]
        ovejas_pos = np.empty(shape=(0, 2), dtype=np.float64)
        if pers_ovejas > 0 and n_sheeps > 0:
            relative_to_sheeps = sheeps - shepherdPosition
            sheepsDists = np.empty(n_sheeps, dtype=np.float64)
            for i in range(n_sheeps):
                sheepsDists[i] = (
                    sheeps[i, 0] * sheeps[i, 0] + sheeps[i, 1] * sheeps[i, 1]
                )
            cercanas_idx = np.empty(pers_ovejas, np.int64)
            for i in range(pers_ovejas):  # esto reemplaza a np.argpartition
                mejor = 0
                for j in range(1, n_sheeps):
                    if sheepsDists[j] < sheepsDists[mejor]:
                        mejor = j
                cercanas_idx[i] = mejor
                sheepsDists[mejor] = np.inf
            ovejas_pos = np.empty((pers_ovejas, 2), dtype=np.float64)
            for i in range(pers_ovejas):
                ovejas_pos[i, 0] = relative_to_sheeps[cercanas_idx[i], 0]
                ovejas_pos[i, 1] = relative_to_sheeps[cercanas_idx[i], 1]

        return ovejas_pos

    @staticmethod
    @jit(nopython=True)
    def pastoresPos(
        shepherds: np.ndarray[np.float64],
        shepherdPosition: np.ndarray[np.float64],
        pers_pastores: int,
    ):
        # NEAREST PASTORES
        n_shepherds = shepherds.shape[0]
        pastores_pos = np.empty(shape=(0, 2), dtype=np.float64)
        if pers_pastores > 0 and n_shepherds > 0:
            relative_to_shepherds = shepherds - shepherdPosition
            shepherdsDists = np.empty(n_shepherds, dtype=np.float64)
            for i in range(n_shepherds):
                shepherdsDists[i] = (
                    shepherds[i, 0] * shepherds[i, 0]
                    + shepherds[i, 1] * shepherds[i, 1]
                )
            cercanos_idx = np.empty(pers_pastores, np.int64)
            for i in range(pers_pastores):  # esto reemplaza a np.argpartition
                mejor = 0
                for j in range(1, n_shepherds):
                    if shepherdsDists[j] < shepherdsDists[mejor]:
                        mejor = j
                cercanos_idx[i] = mejor
                shepherdsDists[mejor] = np.inf
            pastores_pos = np.empty((pers_pastores, 2), dtype=np.float64)
            for i in range(pers_pastores):
                pastores_pos[i, 0] = relative_to_shepherds[cercanos_idx[i], 0]
                pastores_pos[i, 1] = relative_to_shepherds[cercanos_idx[i], 1]

        return pastores_pos


class ShepherdNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim_1=128, hidden_dim_2=64, output_dim=2):
        super().__init__()
        # Arquitectura Napolitano
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, output_dim),
        )

    def forward(self, x):
        return self.net(x)
