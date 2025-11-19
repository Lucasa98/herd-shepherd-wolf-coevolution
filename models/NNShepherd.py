from numba import jit
import torch
import torch.nn as nn
import numpy as np
from sim.shepherd import Shepherd


class NNShepherdModel:
    def __init__(self, params, rng, nn_model):
        self.params = params
        self.rng = rng
        self.nn = nn_model

    def update(
        self,
        shepherd: Shepherd,
        sheeps: np.ndarray[np.float64],
        shepherds: np.ndarray[np.float64],
        objetivo_c: np.ndarray[np.float64],
        centroide: np.ndarray[np.float64],
        diag,
    ):
        """
        Actualiza la posición del pastor según la salida de la red neuronal
        Entradas normalizadas y relativas al pastor
        """
        ovejas_pos = self.ovejasPos(
            shepherdPosition=shepherd.position,
            sheeps=sheeps,
            pers_ovejas=self.params["pers_ovejas"],
            diag=diag,
        )

        pastores_pos = self.pastoresPos(
            shepherds=shepherds,
            shepherdPosition=shepherd.position,
            pers_pastores=self.params["pers_pastores"],
            diag=diag,
        )

        centroide_pos = self.centroidePos(
            shepherdPosition=shepherd.position,
            centroide=centroide,
            diag=diag,
        )

        objetivo_pos = self.objetivoPos(
            shepherdPosition=shepherd.position,
            objetivo=objetivo_c,
            diag=diag,
        )

        # Concatenar: pos rel a centroide + pos rel a vecinos + por rel a ovejas + pos rel a objetivo
        inputs = np.concatenate(
            [
                centroide_pos.ravel(),
                pastores_pos.ravel(),
                ovejas_pos.ravel(),
                objetivo_pos.ravel(),
            ]
        )

        # FORWARD PASS
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
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
        diag,
    ):
        # NEAREST OVEJAS
        n_sheeps = sheeps.shape[0]
        ovejas_pos = np.empty(shape=(0, 2), dtype=np.float64)
        if pers_ovejas > 0 and n_sheeps > 0:
            relative_to_sheeps = sheeps - shepherdPosition
            sheepsDists = np.empty(n_sheeps, dtype=np.float64)
            for i in range(n_sheeps):
                sheepsDists[i] = (
                    relative_to_sheeps[i, 0] * relative_to_sheeps[i, 0]
                    + relative_to_sheeps[i, 1] * relative_to_sheeps[i, 1]
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
                ovejas_pos[i, 0] = relative_to_sheeps[cercanas_idx[i], 0] / diag
                ovejas_pos[i, 1] = relative_to_sheeps[cercanas_idx[i], 1] / diag

        return ovejas_pos

    @staticmethod
    @jit(nopython=True)
    def pastoresPos(
        shepherds: np.ndarray[np.float64],
        shepherdPosition: np.ndarray[np.float64],
        pers_pastores: int,
        diag,
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
                pastores_pos[i, 0] = relative_to_shepherds[cercanos_idx[i], 0] / diag
                pastores_pos[i, 1] = relative_to_shepherds[cercanos_idx[i], 1] / diag

        return pastores_pos

    @staticmethod
    @jit(nopython=True)
    def centroidePos(
        shepherdPosition: np.ndarray[np.float64],
        centroide: np.ndarray[np.float64],
        diag,
    ):
        return (shepherdPosition - centroide) / diag

    @staticmethod
    @jit(nopython=True)
    def objetivoPos(
        shepherdPosition: np.ndarray[np.float64],
        objetivo: np.ndarray[np.float64],
        diag,
    ):
        return (shepherdPosition - objetivo) / diag


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
