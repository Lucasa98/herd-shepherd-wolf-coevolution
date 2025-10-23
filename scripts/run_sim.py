import pygame
from sim.world import World

pygame.init()
screen: pygame.Surface = pygame.display.set_mode((800, 600))
world = World(800, 600)
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