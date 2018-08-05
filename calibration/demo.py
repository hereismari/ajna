import argparse
import json
import pygame
import random

from model import Model
from pygame import gfxdraw

import cv2


parser = argparse.ArgumentParser(description='Webcam')

parser.add_argument('--model-checkpoint', type=str, default='../checkpoints/best_cnn.ckpt')
parser.add_argument('--model-crop-eyes', type=str, default='../shape_predictor_68_face_landmarks.dat', help='download it from: https://drive.google.com/firun_prele/d/1XvAobn_6xeb8Ioa8PBnpCXZm8mgkBTiJ/view?usp=sharing')
parser.add_argument('--eye-shape', type=int, nargs="+", default=[90, 60])
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
    font = pygame.font.Font(None, 72)
    surface = font.render(text, True, (0, 0, 0))

    width = surface.get_width()
    height = surface.get_height()

    display.blit(surface, (x -  width / 2, y - height / 2))


def average(numbers):
    return sum(numbers) / len(numbers)


class CalibrationScreen():
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

    def draw(self, display):
        draw_text(display, '%d/%d' % (self.index, len(self.points)), info.current_w / 2, info.current_h / 2)
        draw_text(display, 'Olhe para o ponto e aperte espaço', info.current_w / 2, info.current_h / 2 + 50)

        if self.index < len(self.points):
            x, y, h, v = self.points[self.index]
            draw_point(display, x, y)

    def react(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                x, y, h, v = self.points[self.index]
                frame, eyes = model.run()
                print(eyes)

                for theta, phi in eyes:
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

            with open('thresholds.txt', 'w') as f:
                text = json.dumps(thresholds)
                f.write(text)

            return None
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

    screen = CalibrationScreen()
    model = Model(args)

    while screen:
        draw()
        delta = clock.tick(10)
        screen = update(delta)

    pygame.quit()
    model.close()
