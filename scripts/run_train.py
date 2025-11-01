import pygame
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, VIDEORESIZE
from numpy.random import default_rng, Generator
from sim.world import World
from sim.interface import Interface

# ============================================================
# ======================== PARAMETROS ========================
# ============================================================
rng: Generator = default_rng()
params = {
    # ESCENARIO
    "w_w": 300,  # ancho
    "w_h": 300,  # alto
    "obj_r": 25,  # radio del objetivo
    # OVEJAS
    "N": 50,  # numero de ovejas
    "r_s": 65.0,  # radio de repulsion del pastor
    "r_a": 2.0,  # radio de repulsion de otras ovejas
    "h": 0.5,  # coeficiente de inercia
    "c": 1.05,  # coeficiente de cohesion
    "rho_a": 2.0,  # fuerza de repulsion (oveja-oveja)
    "rho_s": 1.0,  # fuerza de repulsion (pastor-oveja)
    "e": 0.3,  # ruido angular (componente estocastica)
    "delta": 1.0,  # distancia por paso
    "n_neigh": 7,  # numero de vecinos para cohesion
    "r_walk": 0.05,  # probabilidad de random walk
    # PASTOR
    "p_delta": 1.5,  # distancia por paso
}
# ==========================================================

world = World(params["w_w"], params["w_h"], params, rng)

running = True
while running:
    world.update()
