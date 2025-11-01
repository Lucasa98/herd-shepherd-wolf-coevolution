import pygame


class Interface:
    def __init__(self, font_size=16, color=(255, 255, 255)):
        pygame.font.init()
        self.font = pygame.font.SysFont("Consolas", font_size)
        self.color = color

    def draw(self, surface: pygame.Surface, world):
        lines = [
            f"Ticks: {world.ticks}",
            f"Sheep: {len(world.ovejas)}",
            f"Shepherds: {len(world.pastores)}",
            f"Finished: {world.ticks_to_finish if world.ticks_to_finish else '-'}",
        ]
        for i, text in enumerate(lines):
            rendered = self.font.render(text, True, self.color)
            surface.blit(rendered, (10, 10 + i * 18))
