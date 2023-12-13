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

def calculate_best_move_to_reach_pos(x_pos_des, y_pos_des, x_pos_cur, y_pos_cur):
    # evaluate moves based on distance to pos
    # the the best move is the one that minimizes the distance to the pos
    smallest_dist_to_pos = 999999
    possible_moves = list(Directions)
    best_move = Directions.STAY
    for pmove in possible_moves:
        new_x_pos, new_y_pos = get_new_pos_from_direction(pmove, x_pos_cur, y_pos_cur)

        dist = np.sqrt((x_pos_des - new_x_pos)**2 + (y_pos_des - new_y_pos)**2)

        if dist < smallest_dist_to_pos:
            smallest_dist_to_pos = dist
            best_move = pmove

    return best_move