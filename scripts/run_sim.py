import yaml
import pygame
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, VIDEORESIZE
from numpy.random import default_rng, Generator
from sim.world import World
from sim.interface import Interface

with open("config.yaml") as f:
    params = yaml.safe_load(f)

params["model"] = "followMouse"

rng: Generator = default_rng()

pygame.init()

screen: pygame.Surface = pygame.display.set_mode((600, 600), RESIZABLE)
world_surface: pygame.Surface = pygame.surface.Surface((params["w_w"], params["w_h"]))
world = World(params["w_w"], params["w_h"], params, rng)
interface = Interface()

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
    screen.blit(pygame.transform.scale(world_surface, screen.get_rect().size), (0, 0))
    # Interfaz
    interface.draw(screen, world)
    # ===============================================

    pygame.display.flip()
    dt = clock.tick(60) / 100

pygame.quit()
