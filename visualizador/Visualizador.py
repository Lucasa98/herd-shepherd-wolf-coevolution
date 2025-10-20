import pygame

class Visualizador:
    def __init__(self, num_shepherds, num_sheeps, world_size):
        pygame.init()
        self.screen_size = 600
        self.world_size = world_size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        self.clock = pygame.time.Clock()

    def render(self, shepherds, sheeps, goal):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                import sys
                sys.exit()

        self.screen.fill((0, 0, 0))  # fondo blanco

        # escalar posiciones al tamaño de pantalla
        scale = self.screen_size / self.world_size

        # shepherds (azul)
        for shepherd in shepherds:
            pos = (int(shepherd[0] * scale), int(shepherd[1] * scale))
            pygame.draw.circle(self.screen, (0, 0, 255), pos, 5)

        # sheep (blanco)
        for sheep in sheeps:
            pos = (int(sheep[0] * scale), int(sheep[1] * scale))
            pygame.draw.circle(self.screen, (255, 255, 255), pos, 5)

        # objetivo (verde)
        gpos = (int(goal[0] * scale), int(goal[1] * scale))
        pygame.draw.circle(self.screen, (0, 255, 0), gpos, 8)

        pygame.display.flip()
        self.clock.tick(60)

    def deltatime(self):
        """Devuelve el tiempo entre frames en segundos."""
        return self.clock.get_time() / 1000.0
