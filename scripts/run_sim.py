import pygame
from numpy.random import default_rng, Generator
import numpy as np
from sim.world import World

# ============================================================
# ======================== PARAMETROS ========================
# ============================================================
rng: Generator = default_rng()
params = {
    # OVEJAS
    "r_s": 65.0,    # radio de repulsion del pastor
    "r_a": 2.0,     # radio de repulsion de otras ovejas
    "h": 0.5,       # coeficiente de inercia
    "c": 1.05,      # coeficiente de cohesion
    "rho_a": 2.0,   # fuerza de repulsion (oveja-oveja)
    "rho_s": 1.0,   # fuerza de repulsion (pastor-oveja)
    "e": 0.3,       # ruido angular (componente estocastica)
    "delta": 5.0,   # distancia por paso
    "n_neigh": 4,   # numero de vecinos para cohesion
    "r_walk": 0.01,  # probabilidad de random walk
    # PASTOR
    "p_delta": 3  # distancia por paso
}
# ==========================================================

pygame.init()
screen: pygame.Surface = pygame.display.set_mode((800, 600))
world = World(800, 600, params, rng)
clock = pygame.time.Clock()
dt = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((30, 30, 30))

    # RENDER y UPDATE
    world.update(dt)
    world.draw(screen)
    # ===============

    pygame.display.flip()
    dt = clock.tick(60) / 100

pygame.quit()
