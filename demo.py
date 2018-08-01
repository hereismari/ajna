import pygame
import random

from geometry import Camera, Eye
from pygame import gfxdraw


pygame.init()
pygame.display.set_caption('Ajna')

display = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
info = pygame.display.Info()
clock = pygame.time.Clock()


class Demo:
    def __init__(self, camera):
        self.camera = camera
        self.point = (0, 0)

    def draw(self, display):
        gfxdraw.filled_circle(display, self.point[0], self.point[1], 8, (0, 0, 0))
        gfxdraw.aacircle(display, self.point[0], self.point[1], 8, (0, 0, 0))

    def react(self, event):
        pass

    def update(self, delta):
        # TODO: get data from model
        self.point = self.camera.estimate(None, None)
        return self


class Calibration:
    def __init__(self):
        self.camera = Camera(640, 420, 50)

        self.data = [[], [], [], []]
        self.points = 5 * [
            (info.current_w * 1 / 4, info.current_h * 1 / 4, 0),
            (info.current_w * 3 / 4, info.current_h * 1 / 4, 1),
            (info.current_w * 1 / 4, info.current_h * 3 / 4, 2),
            (info.current_w * 3 / 4, info.current_h * 3 / 4, 3),
        ]

        random.shuffle(self.points)
        self.current = self.points.pop()

    def draw(self, display):
        font = pygame.font.Font(None, 72)
        surface = font.render('%d/20' % (20 - len(self.points)), True, (0, 0, 0))

        display.blit(surface, (info.current_w / 2 - surface.get_width() / 2, info.current_h / 2 - surface.get_height() / 2))

        if self.current:
            gfxdraw.filled_circle(display, self.current[0], self.current[1], 8, (0, 0, 0))
            gfxdraw.aacircle(display, self.current[0], self.current[1], 8, (0, 0, 0))

    def react(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if self.current:
                    # TODO: store data from model
                    self.data[self.current[2]].append((None, None))

                if self.points:
                    self.current = self.points.pop()
                else:
                    self.camera.calibrate(info.current_w, info.current_h, *self.data)
                    self.current = None

    def update(self, delta):
        return self if self.current else Demo(self.camera)



screen = Calibration()


def update(delta):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return None
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return None
        screen.react(event)

    return screen.update(delta)


def draw():
    display.fill(0xffffff)
    screen.draw(display)
    pygame.display.flip()


while screen:
    draw()
    delta = clock.tick(60)
    screen = update(delta)

pygame.quit()
