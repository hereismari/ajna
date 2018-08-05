""" moveCircle.py
    create a blue circle sprite and have it
    follow the mouse"""

import pygame
import random
pygame.init()

# Recupera info da resolucao do monitor
info_monitor = pygame.display.Info()

# Define altura e largura da tela de interface
height = int(info_monitor.current_h * 4 / 5)
width = int(info_monitor.current_w / 3)
screen = pygame.display.set_mode((width, height))


class Circle(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((50, 50))
        self.image.fill((255, 255, 255))
        pygame.draw.circle(self.image, (0, 0, 255), (25, 25), 25, 0)
        self.rect = self.image.get_rect()

    def update(self):
        self.rect.center = pygame.mouse.get_pos()


class Target(pygame.sprite.Sprite):
    def __init__(self, pos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([40, 40])
        self.image.fill((255, 0, 0))
        self.rect = self.image.get_rect()
        self.rect.center = pos

    def update(self, pos):
        self.rect.center = pos


def main():
    pygame.display.set_caption("Ajna")

    BACKGROUND_COLOR = (255, 255, 255)
    background = pygame.Surface(screen.get_size())
    background.fill(BACKGROUND_COLOR)
    screen.blit(background, (0, 0))

    player = Circle()
    player_group = pygame.sprite.Group(player)

    target = Target([100, 200])
    target_group = pygame.sprite.Group()
    target_group.add(target)

    clock = pygame.time.Clock()
    keepGoing = True

    while keepGoing:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keepGoing = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    keepGoing = False

        player_group.clear(screen, background)
        player_group.update()
        player_group.draw(screen)

        target_group.draw(screen)

        hit = pygame.sprite.spritecollide(player, target_group, False)

        if hit:
            x = random.randint(0, width - 50)
            y = random.randint(0, height - 50)
            target_group.clear(screen, background)
            target_group.update((x, y))

        pygame.display.flip()

    # return mouse
    pygame.mouse.set_visible(True)


if __name__ == "__main__":
    main()
