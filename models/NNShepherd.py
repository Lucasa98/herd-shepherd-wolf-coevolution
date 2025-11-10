import torch
import torch.nn as nn
import numpy as np


class NNShepherdModel:
    def __init__(self, params, rng, nn_model):
        self.params = params
        self.rng = rng
        self.nn = nn_model

    def update(self, shepherd, sheeps, shepherds, objetivo_c):
        # Entradas: ovejas mas cercanas + el objetivo + los dos pastores vecinos mas cercanos
        # calcular ovejas mas cercanas
        ovejas_pos = np.array([])
        if self.params["pers_ovejas"] > 0:
            diffs = np.array([s.position - shepherd.position for s in sheeps])
            dists = np.sum(diffs**2, axis=1)
            cercanas_idx = np.argpartition(dists, self.params["pers_ovejas"])[
                : self.params["pers_ovejas"]
            ]
            ovejas_pos = np.array([sheeps[i].position for i in cercanas_idx])

        # calcular pastores mas cercanos
        pastores_pos = np.array([])
        if self.params["pers_pastores"] > 0:
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
            pastores_pos = np.array([sheeps[i].position for i in cercanos_idx])

        inputs = np.concatenate(
            [shepherd.position, pastores_pos.ravel(), ovejas_pos.ravel(), objetivo_c]
        )
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        out = self.nn(inputs).detach().numpy().squeeze()

        # normalize and move
        heading = out / (np.linalg.norm(out) + 1e-8)
        shepherd.heading = heading  # sin inercia. Deberiamos probar con inercia?
        shepherd.position += self.params["p_delta"] * heading


class ShepherdNN(nn.Module):
    # TODO: hacer que no se rompa si le cambias input y output dims
    def __init__(self, input_dim=4, hidden_dim_1=128, hidden_dim_2=64, output_dim=2):
        super().__init__()
        # Napolitano architecture:
        # 6 input neurons, 
        # two hidden layers with 128 and 64 neurons respectively, 
        # both with ReLU activation, and 25 output neurons with linear activation.
        # (ademas tiene otra version mas grande con 512, 256
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, output_dim),
        )

    def forward(self, x):
        return self.net(x)
