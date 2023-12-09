import numpy as np
import os, sys
sys.path.append("../")
from util import Directions
from sandpile import Sandpile
from agents import Agent
import unittest

class TestAgentMoveFuncs(unittest.TestCase):

    def test_move_agent_left_from_middle(self,):
        N_grid = 5 #number of cells per side
        DROP_SAND=False

        #middle position
        X_POS_INIT = N_grid // 2
        Y_POS_INIT = N_grid // 2

        EXPECTED_X_POS = X_POS_INIT - 1
        EXPECTED_Y_POS = Y_POS_INIT
        agent = Agent(x_pos_init=X_POS_INIT, y_pos_init=Y_POS_INIT)

        sandpile = Sandpile(N_grid=N_grid, agent=agent, DROP_SAND=DROP_SAND)

        sandpile.print_grid()
        print(agent.get_agent_pos())
        direction = Directions.LEFT

        sandpile.move_agent_in_direction(direction, agent)
        print(agent.get_agent_pos())
        self.assertEqual((EXPECTED_X_POS, EXPECTED_Y_POS), agent.get_agent_pos())


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestAgentMoveFuncs('test_move_agent_left_from_middle'))
    # suite.addTest(TestAgentMoveFuncs('test_widget_resize'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())