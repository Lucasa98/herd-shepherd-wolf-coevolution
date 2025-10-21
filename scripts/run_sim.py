import pygame
from sim.world import World

pygame.init()
screen = pygame.display.set_mode((800, 600))
world = World(800, 600)
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    world.update(1 / 60)
    screen.fill((30, 30, 30))
    world.draw(screen)
    pygame.display.flip()
    clock.tick(60)
