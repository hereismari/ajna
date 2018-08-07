import argparse
import math
import pygame
import random
import time

from model import Model
from geometry import Camera, Eye
from pygame import gfxdraw
from threading import Thread, Lock

import cv2


parser = argparse.ArgumentParser(description='Webcam')

parser.add_argument('--model-checkpoint', type=str, default='../checkpoints/best_cnn.ckpt')
parser.add_argument('--model-crop-eyes', type=str, default='shape_predictor_68_face_landmarks.dat', help='download it from: https://drive.google.com/firun_prele/d/1XvAobn_6xeb8Ioa8PBnpCXZm8mgkBTiJ/view?usp=sharing')
parser.add_argument('--eye-shape', type=int, nargs="+", default=[90, 60])
parser.add_argument('--heatmap-scale', type=float, default=1)
parser.add_argument('--data-format', type=str, default='NHWC')
parser.add_argument('-src', '--source', dest='video_source', type=int, default=0, help='Device index of the camera.')
parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int, default=2, help='Number of workers.')
parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int, default=1, help='Size of the queue.')

args = parser.parse_args()


class Worker(Thread):
    def __init__(self):
        Thread.__init__(self)

        self.model = Model(args)
        self.lock = Lock()

    def run_step(self):
        frame, data = self.model.run()

        if data is not None:
            if sum(1 for d in data) == 2:
                return frame, [Eye(tuple(map(float, coordinates)), float(radius[0][0]), (-math.sin(phi), -1, math.sin(theta))) for (theta, phi), coordinates, radius in data]

    def dispose(self):
        self.model.close()


class Screen(object):
    def __init__(self, camera=None, worker=None):
        if camera is None:
            self.camera = Camera(640, 420, 50)
        else:
            self.camera = camera
        
        if worker is None:
            self.worker = Worker()
        else:
            self.worker = worker

        self.frame = None

    def display_point(self, display, x, y):
        gfxdraw.filled_circle(display, int(x), int(y), 8, (255, 0, 0))
        gfxdraw.aacircle(display, int(x), int(y), 8, (0, 0, 0))

    def react(self, event):
        pass
    
    def update(self, delta):
        pass
    
    def dispose(self):
        self.worker.dispose()
    
    def draw(self, display):
        if self.frame is not None:
            rows = 300
            cols = 300

            # Resize and rotate frame
            frame = cv2.resize(self.frame, (rows, cols))
            M = cv2.getRotationMatrix2D((cols/2,rows/2), 90, 1)
            frame = cv2.warpAffine(frame, M, (cols,rows))

            # Display frame
            surf = pygame.surfarray.make_surface(frame)
            display.blit(surf, (0, 0))


class DemoScreen(Screen):
    def __init__(self, camera, worker):
        super().__init__(camera, worker)
        self.point = None

    def draw(self, display):
        if self.point:
            x, y = self.point

            x = min(max(0, x), info.current_w)
            y = min(max(0, y), info.current_h)

            self.display_point(display, x, y)
        
        super().draw(display)

    def update(self, delta):
        self.frame, eyes = self.worker.run_step()
        self.point = self.camera.projection(*eyes) if eyes else None
        return self


class CalibrationScreen(Screen):
    def __init__(self):
        super().__init__()

        self.data = [[], [], [], []]
        self.points = 1 * [
            (info.current_w * 1 / 4, info.current_h * 1 / 4, 0),
            (info.current_w * 3 / 4, info.current_h * 1 / 4, 1),
            (info.current_w * 1 / 4, info.current_h * 3 / 4, 2),
            (info.current_w * 3 / 4, info.current_h * 3 / 4, 3),
        ]

        self.total_points = len(self.points)
        self.point_counter = 1

        random.shuffle(self.points)
        self.current = self.points.pop()

    def draw(self, display):
        font = pygame.font.Font(None, 72)
        surface = font.render('%d/%d' % (self.point_counter, self.total_points), True, (0, 0, 0))
        display.blit(surface, (info.current_w / 2 - surface.get_width() / 2, info.current_h / 2 - surface.get_height() / 2))
        surface = font.render('Olhe para o ponto e aperte espaço', True, (0, 0, 0))
        display.blit(surface, (info.current_w / 2 - surface.get_width() / 2, info.current_h / 2 - surface.get_height() / 2 + 50))

        if self.current:
            x, y, q = self.current
            self.display_point(display, x, y)

        super().draw(display)

    def react(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.frame, eyes = self.worker.run_step()
                quadrant = self.current[2]

                if eyes is not None:
                    self.data[quadrant].append(eyes)

                    if self.points:
                        self.current = self.points.pop()
                        self.point_counter += 1
                    else:
                        self.camera.calibrate(info.current_w, info.current_h, *self.data)
                        self.current = None

    def update(self, delta):
        if self.current is None:
            return DemoScreen(self.camera, self.worker)
        return self

    def dispose(self):
        pass


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


if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption('Ajna')

    display = pygame.display.set_mode((0, 0))
    info = pygame.display.Info()
    clock = pygame.time.Clock()

    screen = CalibrationScreen()

    while screen:
        draw()
        delta = clock.tick(1)
        next = update(delta)

        if next is not screen:
            screen.dispose()
            screen = next

    pygame.quit()