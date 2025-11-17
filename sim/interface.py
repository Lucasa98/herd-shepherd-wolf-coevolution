import pygame
import numpy as np

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

            if len(world.pastores) == 1:
                cam_w, cam_h = 150, 150
                shepherd = world.pastores[0]
                view_size = 80
                top_left = shepherd.position - np.array([view_size / 2, view_size / 2])

                scaled_surface = pygame.Surface((view_size, view_size))
                scaled_surface.blit(
                    world._surface,
                    (0, 0),
                    pygame.Rect(top_left[0], top_left[1], view_size, view_size),
                )

                mini = pygame.transform.scale(scaled_surface, (cam_w, cam_h))

                y_text_end = 10 + len(lines) * 18 + 10
                x_pos = surface.get_width() // 2 - cam_w // 2
                y_pos = y_text_end + 10

                surface.blit(mini, (x_pos, y_pos))
                pygame.draw.rect(
                    surface, "white", (x_pos - 2, y_pos - 2, cam_w + 4, cam_h + 4), 2
                )