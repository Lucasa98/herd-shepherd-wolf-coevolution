import pygame
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, VIDEORESIZE
from numpy.random import default_rng, Generator
from sim.world import World

# ============================================================
# ======================== PARAMETROS ========================
# ============================================================
rng: Generator = default_rng()
params = {
    # ESCENARIO
    "w_w": 150,  # ancho
    "w_h": 150,  # alto
    # OVEJAS
    "N": 51,  # numero de ovejas
    "r_s": 65.0,  # radio de repulsion del pastor
    "r_a": 2.0,  # radio de repulsion de otras ovejas
    "h": 0.5,  # coeficiente de inercia
    "c": 1.05,  # coeficiente de cohesion
    "rho_a": 2.0,  # fuerza de repulsion (oveja-oveja)
    "rho_s": 1.0,  # fuerza de repulsion (pastor-oveja)
    "e": 0.3,  # ruido angular (componente estocastica)
    "delta": 1.0,  # distancia por paso
    "n_neigh": 50,  # numero de vecinos para cohesion
    "r_walk": 0.05,  # probabilidad de random walk
    # PASTOR
    "p_delta": 1.5,  # distancia por paso
}
# ==========================================================

pygame.init()
screen: pygame.Surface = pygame.display.set_mode((600, 600), RESIZABLE)
world_surface: pygame.Surface = pygame.surface.Surface((150, 150))
world = World(params["w_w"], params["w_h"], params, rng)
clock = pygame.time.Clock()
dt = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == VIDEORESIZE:
            screen = pygame.display.set_mode(
                event.size, HWSURFACE | DOUBLEBUF | RESIZABLE
            )

    world_surface.fill((30, 30, 30))

    # =============== RENDER y UPDATE ===============
    # Simulacion
    world.update()
    world.draw(world_surface)
    # Interfaz
    # ===============================================

    screen.blit(pygame.transform.scale(world_surface, screen.get_rect().size), (0, 0))
    pygame.display.flip()
    dt = clock.tick(60) / 100

pygame.quit()
