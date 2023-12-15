import numpy as np
import os, sys
sys.path.append("../")
from util import Directions, calculate_best_move_to_reach_pos
import unittest

class TestCalcBestMoveReachPos(unittest.TestCase):

    def test_calculate_best_move_to_reach_pos_left(self,):
        x_pos_cur = 0
        y_pos_cur = 0
        x_pos_des = -10
        y_pos_des = 0

        move = calculate_best_move_to_reach_pos(x_pos_des, y_pos_des, x_pos_cur, y_pos_cur)
        print(move)
        self.assertTrue(move, Directions.LEFT)

    def test_calculate_best_move_to_reach_pos_right(self,):
        x_pos_cur = 0
        y_pos_cur = 0
        x_pos_des = 10
        y_pos_des = 0

        move = calculate_best_move_to_reach_pos(x_pos_des, y_pos_des, x_pos_cur, y_pos_cur)
        print(move)
        self.assertTrue(move, Directions.RIGHT)

    def test_calculate_best_move_to_reach_pos_up(self,):
        x_pos_cur = 0
        y_pos_cur = 0
        x_pos_des = 0
        y_pos_des = -10

        move = calculate_best_move_to_reach_pos(x_pos_des, y_pos_des, x_pos_cur, y_pos_cur)
        print(move)
        self.assertTrue(move, Directions.UP)

    def test_calculate_best_move_to_reach_pos_down(self,):
        x_pos_cur = 0
        y_pos_cur = 0
        x_pos_des = 0
        y_pos_des = 10

        move = calculate_best_move_to_reach_pos(x_pos_des, y_pos_des, x_pos_cur, y_pos_cur)
        print(move)
        self.assertTrue(move, Directions.DOWN)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestCalcBestMoveReachPos('test_calculate_best_move_to_reach_pos_left'))
    suite.addTest(TestCalcBestMoveReachPos('test_calculate_best_move_to_reach_pos_right'))
    suite.addTest(TestCalcBestMoveReachPos('test_calculate_best_move_to_reach_pos_up'))
    suite.addTest(TestCalcBestMoveReachPos('test_calculate_best_move_to_reach_pos_down'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    results = runner.run(suite())
    print(results)