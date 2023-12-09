import numpy as np
import os, sys
sys.path.append("../")
from util import Directions
from sandpile import Sandpile
from agents import MaxAgent
import unittest

class TestMaxAgentMoveFuncsFromMiddle(unittest.TestCase):

    def setUp(self):
        #middle position
        self.N_grid = 5 #number of cells per side
        
    def setup_agent_grid(self, X_POS_INIT, Y_POS_INIT, N_grid):
        DROP_SAND=False
        agent = MaxAgent(x_pos_init=X_POS_INIT, y_pos_init=Y_POS_INIT)
        sandpile = Sandpile(N_grid=N_grid, agents=[agent], DROP_SAND=DROP_SAND)
        return agent, sandpile

    def test_maxAgent_chooses_single_max_left(self,):

        
        X_POS_INIT = self.N_grid//2
        Y_POS_INIT = self.N_grid//2
        
        agent, sandpile = self.setup_agent_grid(X_POS_INIT, Y_POS_INIT, self.N_grid)

        # set up grid to have maximum at left of agent
        EXPECTED_X_POS = X_POS_INIT - 1
        EXPECTED_Y_POS = Y_POS_INIT
        sandpile.grid[EXPECTED_Y_POS, EXPECTED_X_POS] = 3
        EXPECTED_DIRECTION = Directions.LEFT

        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())

        # get move from maxAgent
        direction = agent.choose_move(sandpile)
        print(direction)

        agent.move_agent_in_direction(direction)
        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())
        self.assertEqual((EXPECTED_Y_POS, EXPECTED_X_POS), agent.get_agent_pos())
        self.assertEqual(direction, EXPECTED_DIRECTION)

    def test_maxAgent_chooses_single_max_up(self,):

        
        X_POS_INIT = self.N_grid//2
        Y_POS_INIT = self.N_grid//2
        
        agent, sandpile = self.setup_agent_grid(X_POS_INIT, Y_POS_INIT, self.N_grid)

        # set up grid to have maximum above agent
        EXPECTED_X_POS = X_POS_INIT
        EXPECTED_Y_POS = Y_POS_INIT - 1
        sandpile.grid[EXPECTED_Y_POS, EXPECTED_X_POS] = 3
        EXPECTED_DIRECTION = Directions.UP
        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())

        # get move from maxAgent
        direction = agent.choose_move(sandpile)
        print(direction)

        agent.move_agent_in_direction(direction)
        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())
        self.assertEqual((EXPECTED_Y_POS, EXPECTED_X_POS), agent.get_agent_pos())
        self.assertEqual(direction, EXPECTED_DIRECTION)

    def test_maxAgent_chooses_single_max_stay(self,):

        
        X_POS_INIT = self.N_grid//2
        Y_POS_INIT = self.N_grid//2
        
        
        agent, sandpile = self.setup_agent_grid(X_POS_INIT, Y_POS_INIT, self.N_grid)

        # set up grid to have maximum at agent pos
        EXPECTED_X_POS = X_POS_INIT
        EXPECTED_Y_POS = Y_POS_INIT
        sandpile.grid[EXPECTED_Y_POS, EXPECTED_X_POS] = 3
        EXPECTED_DIRECTION = Directions.STAY
        
        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())

        # get move from maxAgent
        direction = agent.choose_move(sandpile)
        print(direction)

        agent.move_agent_in_direction(direction)
        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())
        self.assertEqual((EXPECTED_Y_POS, EXPECTED_X_POS), agent.get_agent_pos())
        self.assertEqual(direction, EXPECTED_DIRECTION)

    def test_maxAgent_chooses_from_two_max(self,):

        
        X_POS_INIT = self.N_grid//2
        Y_POS_INIT = self.N_grid//2
        
        
        agent, sandpile = self.setup_agent_grid(X_POS_INIT, Y_POS_INIT, self.N_grid)

        # set up two maxes, one above agent, one to thr right of agent
        EXPECTED_X_POS1, EXPECTED_Y_POS1 = X_POS_INIT, Y_POS_INIT - 1
        EXPECTED_X_POS2, EXPECTED_Y_POS2 = X_POS_INIT + 1, Y_POS_INIT
        EXPECTED_DIRECTIONS = [Directions.UP, Directions.RIGHT]
        sandpile.grid[EXPECTED_Y_POS1, EXPECTED_X_POS1] = 3
        sandpile.grid[EXPECTED_Y_POS2, EXPECTED_X_POS2] = 3

        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())

        # get move from maxAgent
        direction = agent.choose_move(sandpile)
        print(direction)

        agent.move_agent_in_direction(direction)
        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())

        self.assertIn(direction, EXPECTED_DIRECTIONS)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestMaxAgentMoveFuncsFromMiddle('test_maxAgent_chooses_single_max_left'))
    suite.addTest(TestMaxAgentMoveFuncsFromMiddle('test_maxAgent_chooses_single_max_up'))
    suite.addTest(TestMaxAgentMoveFuncsFromMiddle('test_maxAgent_chooses_single_max_stay'))
    suite.addTest(TestMaxAgentMoveFuncsFromMiddle('test_maxAgent_chooses_from_two_max'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    results = runner.run(suite())
    print(results)