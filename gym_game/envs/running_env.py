import pygame
from pygame.locals import QUIT
import sys

# "main"
def pygame_events(space, myenv):
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()