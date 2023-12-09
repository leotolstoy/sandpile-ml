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
        EXPECTED_X_POS = X_POS_INIT - 1
        EXPECTED_Y_POS = Y_POS_INIT
        
        agent, sandpile = self.setup_agent_grid(X_POS_INIT, Y_POS_INIT, self.N_grid)

        # set up grid to have maximum at left of agent
        sandpile.grid[EXPECTED_Y_POS, EXPECTED_X_POS] = 3

        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())

        # get move from maxAgent
        direction = agent.choose_move(sandpile)
        print(direction)

        sandpile.move_agent_in_direction(direction, agent)
        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())
        self.assertEqual((EXPECTED_Y_POS, EXPECTED_X_POS), agent.get_agent_pos())
        self.assertEqual(direction, Directions.LEFT)

    def test_maxAgent_chooses_single_max_up(self,):

        
        X_POS_INIT = self.N_grid//2
        Y_POS_INIT = self.N_grid//2
        EXPECTED_X_POS = X_POS_INIT
        EXPECTED_Y_POS = Y_POS_INIT - 1
        
        agent, sandpile = self.setup_agent_grid(X_POS_INIT, Y_POS_INIT, self.N_grid)

        # set up grid to have maximum above agent
        sandpile.grid[EXPECTED_Y_POS, EXPECTED_X_POS] = 3

        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())

        # get move from maxAgent
        direction = agent.choose_move(sandpile)
        print(direction)

        sandpile.move_agent_in_direction(direction, agent)
        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())
        self.assertEqual((EXPECTED_Y_POS, EXPECTED_X_POS), agent.get_agent_pos())
        self.assertEqual(direction, Directions.UP)

    def test_maxAgent_chooses_single_max_stay(self,):

        
        X_POS_INIT = self.N_grid//2
        Y_POS_INIT = self.N_grid//2
        EXPECTED_X_POS = X_POS_INIT
        EXPECTED_Y_POS = Y_POS_INIT
        
        agent, sandpile = self.setup_agent_grid(X_POS_INIT, Y_POS_INIT, self.N_grid)

        # set up grid to have maximum above agent
        sandpile.grid[EXPECTED_Y_POS, EXPECTED_X_POS] = 3

        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())

        # get move from maxAgent
        direction = agent.choose_move(sandpile)
        print(direction)

        sandpile.move_agent_in_direction(direction, agent)
        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())
        self.assertEqual((EXPECTED_Y_POS, EXPECTED_X_POS), agent.get_agent_pos())
        self.assertEqual(direction, Directions.STAY)

    def test_maxAgent_chooses_from_two_max(self,):

        
        X_POS_INIT = self.N_grid//2
        Y_POS_INIT = self.N_grid//2
        EXPECTED_X_POS = X_POS_INIT
        EXPECTED_Y_POS = Y_POS_INIT - 1
        
        agent, sandpile = self.setup_agent_grid(X_POS_INIT, Y_POS_INIT, self.N_grid)

        # set up grid to have maximum above agent
        sandpile.grid[EXPECTED_Y_POS, EXPECTED_X_POS] = 3

        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())

        # get move from maxAgent
        direction = agent.choose_move(sandpile)
        print(direction)

        sandpile.move_agent_in_direction(direction, agent)
        sandpile.print_grid_and_agent_pos(agent)
        print(agent.get_agent_pos())
        self.assertEqual((EXPECTED_Y_POS, EXPECTED_X_POS), agent.get_agent_pos())
        self.assertEqual(direction, Directions.UP)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestMaxAgentMoveFuncsFromMiddle('test_maxAgent_chooses_single_max_left'))
    suite.addTest(TestMaxAgentMoveFuncsFromMiddle('test_maxAgent_chooses_single_max_up'))
    suite.addTest(TestMaxAgentMoveFuncsFromMiddle('test_maxAgent_chooses_single_max_stay'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())