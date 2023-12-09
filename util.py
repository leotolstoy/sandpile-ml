import numpy as np
from enum import Enum
import random

class Directions(Enum):
    STAY = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4

def get_new_pos_from_direction(direction, x_pos, y_pos):
    if direction == Directions.LEFT:
        new_x_pos = x_pos - 1
        new_y_pos = y_pos

    elif direction == Directions.RIGHT:
        new_x_pos = x_pos + 1
        new_y_pos = y_pos

    elif direction == Directions.UP:
        new_x_pos = x_pos
        new_y_pos = y_pos - 1

    elif direction == Directions.DOWN:
        new_x_pos = x_pos
        new_y_pos = y_pos + 1

    elif direction == Directions.STAY:
        new_x_pos = x_pos
        new_y_pos = y_pos

    return new_x_pos, new_y_pos

def choose_random_neighbor_from_point(x_pos, y_pos):

    possible_moves = [Directions.LEFT, Directions.RIGHT, Directions.UP, Directions.DOWN]
    direction = random.choice(possible_moves)
    new_x_pos, new_y_pos = get_new_pos_from_direction(direction, x_pos, y_pos)

    return new_x_pos, new_y_pos