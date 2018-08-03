# coding: utf-8

import math

from sympy import Point, Line, Plane, Rational


zero = Point(dim=2)

abscissa = Line((0, 0), (1, 0))
ordinate = Line((0, 0), (0, 1))


# Calcula o produto vetorial
def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


class Eye:
    def __init__(self, coordinates, radius, gaze):
        self.coordinates = coordinates
        self.radius = radius
        self.gaze = gaze


class Camera:
    def __init__(self, width, height, fov):
        self.zero = Point(-width, width / math.tan(math.pi * fov / 360), -height) / 2

    # Recebe a resolução do monitor e listas de pares de olhos olhando para o centro de cada quadrante dele
    def calibrate(self, width, height, top_left, top_right, bottom_left, bottom_right):
        top_left = self.estimate_average(top_left)
        top_right = self.estimate_average(top_right)
        bottom_left = self.estimate_average(bottom_left)
        bottom_right = self.estimate_average(bottom_right)

        vertical_left = Line(bottom_left, top_left)
        vertical_right = Line(bottom_right, top_right)

        horizontal_top = Line(top_left, top_right)
        horizontal_bottom = Line(bottom_left, bottom_right)

        self.abscissa_hinge, = vertical_left.intersection(vertical_right)
        self.ordinate_hinge, = horizontal_top.intersection(horizontal_bottom)

        abscissa_low, = vertical_left.intersection(abscissa)
        abscissa_high, = vertical_right.intersection(abscissa)

        ordinate_low, = horizontal_bottom.intersection(ordinate)
        ordinate_high, = horizontal_top.intersection(ordinate)

        self.to_width = lambda x: width / 4 + width / 2 * (x - abscissa_low.x) / (abscissa_high.x - abscissa_low.x)
        self.to_height = lambda y: height / 4 + height / 2 * (y - ordinate_low.y) / (ordinate_high.y - ordinate_low.y)

    # Calcula para onde no monitor se está olhando
    def projection(self, eye1, eye2):
        target = self.estimate(eye1, eye2)

        horizontal, = Line(target, self.abscissa_hinge).intersection(abscissa)
        vertical, = Line(target, self.ordinate_hinge).intersection(ordinate)

        x = self.to_width(horizontal.x)
        y = self.to_height(vertical.y)

        return float(x), float(y)

    # Traça uma reta da camera até o ponto no monitor para onde se está olhando
    def estimate(self, eye1, eye2):
        point1 = self.to_3D(eye1.coordinates)
        point2 = self.to_3D(eye2.coordinates) * Rational(eye1.radius / eye2.radius)

        normal = cross(eye1.gaze, eye2.gaze)

        plane1 = Plane(point1, normal_vector=normal)
        plane2 = Plane(point2, normal_vector=normal)

        line1 = Line(point1, direction_ratio=eye1.gaze)
        line2 = Line(point2, direction_ratio=eye2.gaze)

        plane1_line1 = plane1.projection_line(line1)
        plane1_line2 = plane1.projection_line(line2)
        plane2_line1 = plane2.projection_line(line1)
        plane2_line2 = plane2.projection_line(line2)

        target1, = plane1_line1.intersection(plane1_line2)
        target2, = plane2_line1.intersection(plane2_line2)

        return self.to_2D(target1 + target2) / 2

    # Faz a média do ponto para onde se está olhando usando muitos dados
    def estimate_average(self, array):
        results = [self.estimate(eye1, eye2) for eye1, eye2 in array]
        return sum(results, zero) / len(results)

    # Converte um ponto 3D para uma coordenada na webcam
    def to_2D(self, point):
        return Point(point.x, point.z) / point.y

    # Converte uma coordenada na imagem da webcam para um ponto 3D
    def to_3D(self, point):
        x, y = point
        return self.zero + Point(x, 0, y)
