import pygame
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, VIDEORESIZE
from sim.world import World
from sim.interface import Interface
from numpy.random import default_rng

class Window:
    def __init__(self, params):
        self.params = params
        self.rng = default_rng()

        pygame.init()

        self.interface_width = 250
        self.screen_width = params["w_w"] + self.interface_width
        self.screen_height = params["w_h"]

        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height), HWSURFACE | DOUBLEBUF | RESIZABLE
        )

        self.world_surface = pygame.Surface((params["w_w"], params["w_h"]))
        self.world = World(params["w_w"], params["w_h"], params, self.rng)
        self.interface = Interface()

        self.params["world_scale_x"] = 1.0
        self.params["world_scale_y"] = 1.0
        self.params["world_offset_x"] = 0.0
        self.params["world_offset_y"] = 0.0

        self.clock = pygame.time.Clock()
        self.running = True

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == VIDEORESIZE:
                # mantiene relación horizontal: interfaz fija a la derecha
                self.screen = pygame.display.set_mode(
                    event.size, HWSURFACE | DOUBLEBUF | RESIZABLE
                )

    def update(self):
        self.world.update()

    def render(self):
        self.world_surface.fill((30, 30, 30))
        self.world.draw(self.world_surface)

        screen_w, screen_h = self.screen.get_size()

        available_w = screen_w - self.interface_width
        available_h = screen_h
        scale_factor = min(
            available_w / self.params["w_w"], available_h / self.params["w_h"]
        )

        scaled_w = int(self.params["w_w"] * scale_factor)
        scaled_h = int(self.params["w_h"] * scale_factor)

        scaled_world = pygame.transform.smoothscale(
            self.world_surface, (scaled_w, scaled_h)
        )

        self.screen.fill((0, 0, 0))

        world_x = 0
        world_y = (screen_h - scaled_h) // 2

        self.params["world_scale_x"] = scaled_w / self.params["w_w"]
        self.params["world_scale_y"] = scaled_h / self.params["w_h"]
        self.params["world_offset_x"] = float(world_x)
        self.params["world_offset_y"] = float(world_y)

        self.screen.blit(scaled_world, (world_x, world_y))

        base_interface = pygame.Surface((self.interface_width, self.params["w_h"]))
        base_interface.fill((0, 0, 0))
        self.world._surface = self.world_surface
        self.interface.draw(base_interface, self.world)

        scaled_interface_w = int(self.interface_width * self.params["world_scale_x"])
        scaled_interface_h = int(self.params["w_h"] * self.params["world_scale_y"])

        scaled_interface = pygame.transform.smoothscale(
            base_interface, (scaled_interface_w, scaled_interface_h)
        )

        interface_x = scaled_w
        interface_y = world_y
        self.screen.blit(scaled_interface, (interface_x, interface_y))

        pygame.display.flip()
        self.clock.tick(60)

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.render()
        pygame.quit()
