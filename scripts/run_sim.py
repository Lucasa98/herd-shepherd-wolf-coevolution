import pygame
from sim.world import World

# ============================================================
# ======================== PARAMETROS ========================
# ============================================================
params = {
    # OVEJAS
    "r_s": 65.0,  # radio de repulsion del pastor
    "r_a": 2.0,  # radio de repulsion de otras ovejas
    "h": 0.5,  # coeficiente de inercia
    "c": 1.05,  # coeficiente de cohesion
    "rho_a": 2.0,  # fuerza de repulsion (oveja-oveja)
    "rho_s": 1.0,  # fuerza de repulsion (pastor-oveja)
    "e": 0.3,  # ruido angular (componente estocastica)
    "delta": 1.0,  # distancia por paso
    "n_neigh": 5,  # numero de vecinos para cohesion
}
# ==========================================================

pygame.init()
screen: pygame.Surface = pygame.display.set_mode((800, 600))
world = World(800, 600, params)
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
