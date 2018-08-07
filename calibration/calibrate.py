import argparse
import cv2
import os
import pickle
import pygame
import random

from model import Model
from pygame import gfxdraw


parser = argparse.ArgumentParser(description='Webcam')

parser.add_argument('--model-checkpoint', type=str, default='../checkpoints/best_cnn.ckpt')
parser.add_argument('--model-crop-eyes', type=str, default='../shape_predictor_68_face_landmarks.dat', help='download it from: https://drive.google.com/firun_prele/d/1XvAobn_6xeb8Ioa8PBnpCXZm8mgkBTiJ/view?usp=sharing')
parser.add_argument('--eye-shape', type=int, nargs="+", default=[60, 90])
parser.add_argument('--heatmap-scale', type=float, default=1)
parser.add_argument('--data-format', type=str, default='NHWC')
parser.add_argument('-src', '--source', dest='video_source', type=int, default=0, help='Device index of the camera.')
parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int, default=2, help='Number of workers.')
parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int, default=1, help='Size of the queue.')

args = parser.parse_args()


def draw_point(display, x, y):
    gfxdraw.filled_circle(display, int(x), int(y), 8, (0, 0, 0))
    gfxdraw.aacircle(display, int(x), int(y), 8, (0, 0, 0))


def draw_text(display, text, x, y):
    font = pygame.font.Font(None, 48)
    surface = font.render(text, True, (0, 0, 0))

    width = surface.get_width()
    height = surface.get_height()

    display.blit(surface, (x -  width / 2, y - height / 2))


def draw_frame(display, frame, x, y, size):
    h, w = frame.shape[:2]

    square = max(h, w)
    frame = cv2.resize(frame, (square, square))

    M = cv2.getRotationMatrix2D((square / 2, square / 2), 90, 1)
    frame = cv2.warpAffine(frame, M, (square, square))

    w, h = int(h * size), int(w * size)
    frame = cv2.resize(frame, (w, h))

    surface = pygame.surfarray.make_surface(frame)
    w, h = surface.get_width(), surface.get_height()

    display.blit(surface, (x - w / 2, y - h / 2))


def average(numbers):
    if len(numbers) == 0:
        return None
    return sum(numbers) / len(numbers)


class DemoScreen:
    def __init__(self, thresholds):
        self.thresholds = thresholds

        self.gaze = None
        self.frame = None

    def draw(self, display):
        if self.frame is not None:
            draw_frame(display, self.frame, info.current_w / 2, info.current_h / 2, 0.50)

        if self.gaze is not None:
            theta, phi = self.gaze

            if phi < self.thresholds['left']:
                display.fill(0x42f462, ((0, 0), (info.current_w / 4, info.current_h)))
            elif phi > self.thresholds['right']:
                display.fill(0x42f462, ((info.current_w * 3 / 4, 0), (info.current_w, info.current_h)))

            if theta < self.thresholds['down']:
                display.fill(0x42f462, ((0, info.current_h * 3 / 4), (info.current_w, info.current_h)))
            elif theta > self.thresholds['up']:
                display.fill(0x42f462, ((0, 0), (info.current_w, info.current_h / 4)))

    def react(self, event):
        pass

    def update(self, delta):
        self.frame, eyes = model.run()
        if eyes:
            self.gaze = average(tuple(theta for theta, phi in eyes)), average(tuple(phi for theta, phi in eyes))

        return self


class CalibrationScreen:
    def __init__(self):
        self.horizontal_data = [[], []]
        self.vertical_data = [[], []]

        self.points = 5 * [
            (info.current_w * 1 / 4, info.current_h * 1 / 4, 0, 0),
            (info.current_w * 3 / 4, info.current_h * 1 / 4, 1, 0),
            (info.current_w * 1 / 4, info.current_h * 3 / 4, 0, 1),
            (info.current_w * 3 / 4, info.current_h * 3 / 4, 1, 1),
        ]

        random.shuffle(self.points)

        self.index = 0
        self.frame = None

    def draw(self, display):
        if self.frame is not None:
            draw_frame(display, self.frame, info.current_w / 2, info.current_h / 2, 0.50)

        draw_text(display, '%d/%d' % (self.index, len(self.points)), 96, 96)
        draw_text(display, 'Olhe para o ponto e aperte espaço', info.current_w / 2, info.current_h - 96)

        if self.index < len(self.points):
            x, y, h, v = self.points[self.index]
            draw_point(display, x, y)

    def react(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                x, y, h, v = self.points[self.index]

                for theta, phi in self.eyes:
                    self.horizontal_data[h].append(phi)
                    self.vertical_data[v].append(theta)

                self.index += 1

    def update(self, delta):
        if self.index == len(self.points):
            left, right = map(average, self.horizontal_data)
            up, down = map(average, self.vertical_data)

            thresholds = {
                'left': left,
                'right': right,
                'up': up,
                'down': down,
            }

            with open('../thresholds.pickle', 'wb') as f:
                pickle.dump(thresholds, f)

            return DemoScreen(thresholds)
        else:
            self.frame, self.eyes = model.run()
        return self


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

    display = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    info = pygame.display.Info()
    clock = pygame.time.Clock()

    if os.path.exists('../thresholds.pickle'):
        with open('../thresholds.pickle', 'rb') as f:
            thresholds = pickle.load(f)
        screen = DemoScreen(thresholds)
    else:
        screen = CalibrationScreen()
    model = Model(args)

    while screen:
        draw()
        delta = clock.tick(10)
        screen = update(delta)

    pygame.quit()
    model.close()
