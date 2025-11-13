import torch
import torch.nn as nn
import numpy as np
from sim.shepherd import Shepherd


class NNShepherdModel:
    def __init__(self, params, rng, nn_model):
        self.params = params
        self.rng = rng
        self.nn = nn_model

    def update(self, shepherd: Shepherd, sheeps, shepherds, objetivo_c):
        """
        Actualiza la posición del pastor según la salida de la red neuronal
        Entradas normalizadas y relativas al pastor
        """
        # ENTORNO
        w = float(self.params["w_w"])
        h = float(self.params["w_h"])
        max_dim = max(w, h)

        # NEAREST OVEJAS
        ovejas_pos = np.array([])
        if self.params["pers_ovejas"] > 0 and len(sheeps) > 0:
            diffs = np.array([s.position - shepherd.position for s in sheeps])
            dists = np.sum(diffs**2, axis=1)
            cercanas_idx = np.argpartition(dists, self.params["pers_ovejas"])[
                : self.params["pers_ovejas"]
            ]
            ovejas_pos = np.array([sheeps[i].position for i in cercanas_idx])

        # NEAREST PASTORES
        pastores_pos = np.array([])
        if self.params["pers_pastores"] > 0 and len(shepherds) > 1:
            diffs = np.array(
                [
                    other.position - shepherd.position
                    for other in shepherds
                    if other is not shepherd
                ]
            )
            dists = np.sum(diffs**2, axis=1)
            cercanos_idx = np.argpartition(dists, self.params["pers_pastores"])[
                : self.params["pers_pastores"]
            ]
            pastores_pos = np.array([diffs[i] for i in cercanos_idx])

        # INPUTS RELATIVOS AL PASTOR
        rel_ovejas = (
            ovejas_pos - shepherd.position if ovejas_pos.size > 0 else np.array([])
        )
        rel_pastores = (
            pastores_pos - shepherd.position if pastores_pos.size > 0 else np.array([])
        )
        rel_objetivo = objetivo_c - shepherd.position

        # Concatenar: dirección actual + vecinos + ovejas + objetivo relativo
        inputs = np.concatenate(
            [shepherd.heading, rel_pastores.ravel(), rel_ovejas.ravel(), rel_objetivo]
        )

        # NORMALIZAR INPUTS
        coords = inputs.reshape(-1, 2) if inputs.size % 2 == 0 else None
        if coords is not None:
            coords[:, 0] = (coords[:, 0]) / (w / 2)
            coords[:, 1] = (coords[:, 1]) / (h / 2)
            inputs_norm = np.clip(coords.ravel(), -1.0, 1.0)
        else:
            inputs_norm = np.clip(inputs / (max_dim / 2), -1.0, 1.0)

        # FORWARD PASS
        inputs_tensor = torch.tensor(inputs_norm, dtype=torch.float32).unsqueeze(0)
        out = self.nn(inputs_tensor).detach().numpy().squeeze()

        # MOVER PASTOR
        shepherd.heading = out / (np.linalg.norm(out) + 1e-8)
        # mover
        shepherd.prev_pos = shepherd.position.copy()
        shepherd.position += self.params["p_delta"] * shepherd.heading


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
