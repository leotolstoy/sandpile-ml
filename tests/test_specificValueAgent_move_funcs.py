import numpy as np
import os, sys
sys.path.append("../")
from util import Directions
from sandpile import Sandpile
from agents import SeekSpecificValueAgent
import unittest

class TestSSVAgentAgentMoveFuncsFromMiddle(unittest.TestCase):

    def setUp(self):
        #middle position
        self.N_grid = 5 #number of cells per side
        self.TARGET_VALUE = 1
        
    def setup_agent_grid(self, X_POS_INIT, Y_POS_INIT, N_grid):
        DROP_SAND=False
        agent = SeekSpecificValueAgent(x_pos_init=X_POS_INIT, y_pos_init=Y_POS_INIT, specific_value=self.TARGET_VALUE)
        sandpile = Sandpile(N_grid=N_grid, agents=[agent], DROP_SAND=DROP_SAND)
        return agent, sandpile

    def test_SSVAgent_chooses_single_target(self,):
        
        X_POS_INIT = 2
        Y_POS_INIT = 2
        
        agent, sandpile = self.setup_agent_grid(X_POS_INIT, Y_POS_INIT, self.N_grid)

        # define grid to have targeted value
        # [[0. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]
        # [0. 1. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]]

        # set up grid to have maximum at left of agent
        EXPECTED_X_POS = X_POS_INIT - 1
        EXPECTED_Y_POS = Y_POS_INIT
        sandpile.grid[EXPECTED_Y_POS, EXPECTED_X_POS] = self.TARGET_VALUE
        EXPECTED_DIRECTION = Directions.LEFT

        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())

        # get move from SSVAgent
        direction = agent.choose_move(sandpile)
        print(direction)

        agent.move_agent_in_direction(direction)
        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())
        self.assertEqual((EXPECTED_Y_POS, EXPECTED_X_POS), agent.get_agent_pos())
        self.assertEqual(direction, EXPECTED_DIRECTION)

    def test_SSVAgent_chooses_target_from_two_specific_values(self,):
        
        X_POS_INIT = 2
        Y_POS_INIT = 2
        
        agent, sandpile = self.setup_agent_grid(X_POS_INIT, Y_POS_INIT, self.N_grid)

        # define grid to have targeted value
        # [[0. 0. 0. 0. 0.]
        # [0. 0. 1. 0. 0.]
        # [0. 1. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]]

        # set up two maxes, one above agent, one to thr right of agent
        EXPECTED_X_POS1, EXPECTED_Y_POS1 = X_POS_INIT, Y_POS_INIT - 1
        EXPECTED_X_POS2, EXPECTED_Y_POS2 = X_POS_INIT - 1, Y_POS_INIT
        EXPECTED_DIRECTIONS = [Directions.UP, Directions.LEFT]
        sandpile.grid[EXPECTED_Y_POS1, EXPECTED_X_POS1] = self.TARGET_VALUE
        sandpile.grid[EXPECTED_Y_POS2, EXPECTED_X_POS2] = self.TARGET_VALUE


        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())

        # get move from SSVAgent
        direction = agent.choose_move(sandpile)
        print(direction)

        agent.move_agent_in_direction(direction)
        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())
        self.assertIn(direction, EXPECTED_DIRECTIONS)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestSSVAgentAgentMoveFuncsFromMiddle('test_SSVAgent_chooses_single_target'))
    suite.addTest(TestSSVAgentAgentMoveFuncsFromMiddle('test_SSVAgent_chooses_target_from_two_specific_values'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    results = runner.run(suite())
    print(results)