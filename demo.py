import pygame
import random
import time

from geometry import Camera, Eye
from pygame import gfxdraw
from threading import Thread, Lock


pygame.init()
pygame.display.set_caption('Ajna')

display = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
info = pygame.display.Info()
clock = pygame.time.Clock()


class Model(Thread):
    def __init__(self):
        Thread.__init__(self)

        self.lock = Lock()
        self.data = []
        self.stop = False

    def run(self):
        while not self.stop:
            data = self.run_step()
            now = time.time()

            if data:
                with self.lock:
                    self.data.append((data, now))

    def run_step(self):
        # TODO: get data from model
        return None

    def get(self):
        with self.lock:
            result = self.data
            self.data = []
        return result


class DemoScreen:
    def __init__(self, camera, model):
        self.camera = camera
        self.model = model
        self.points = []

        self.model.start()

    def draw(self, display):
        if self.points:
            pool = len(self.points)

            x = sum(x for (x, y), t in self.points) / pool
            y = sum(y for (x, y), t in self.points) / pool

            gfxdraw.filled_circle(display, x, y, 8, (0, 0, 0))
            gfxdraw.aacircle(display, x, y, 8, (0, 0, 0))

    def react(self, event):
        pass

    def update(self, delta):
        now = time.time()

        for eyes, timestamp in self.model.get():
            point = self.camera.estimate(*eyes)
            self.points.append((point, timestamp))

        self.points = [(p, t) for p, t in self.points if t > now - 1]

        return self

    def dispose(self):
        self.model.stop = True


class CalibrationScreen:
    def __init__(self):
        self.camera = Camera(640, 420, 50)
        self.model = Model()

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
                eyes = self.model.run_step()
                quadrant = self.current[2]

                if eyes is not None:
                    self.data[quadrant].append(eyes)

                    if self.points:
                        self.current = self.points.pop()
                    else:
                        self.camera.calibrate(info.current_w, info.current_h, * self.data)
                        self.current = None

    def update(self, delta):
        if self.current is None:
            return DemoScreen(self.camera, self.model)
        return self

    def dispose(self):
        pass


screen = CalibrationScreen()


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
    delta = clock.tick(30)
    next = update(delta)

    if next is not screen:
        screen.dispose()
        screen = next

pygame.quit()
