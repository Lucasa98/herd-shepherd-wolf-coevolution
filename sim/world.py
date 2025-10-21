#import pygame


class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.entities = []

    def update(self, dt):
        for e in self.entities:
            e.update(dt)

    def draw(self, surface):
        for e in self.entities:
            e.draw(surface)
