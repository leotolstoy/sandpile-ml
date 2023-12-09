import numpy as np
import os, sys
sys.path.append("../")
from util import Directions
from sandpile import Sandpile
from agents import Agent
import unittest

class TestAgentMoveFuncsFromMiddle(unittest.TestCase):

    def setUp(self):
        #middle position
        self.N_grid = 5 #number of cells per side
        self.X_POS_INIT = self.N_grid // 2
        self.Y_POS_INIT = self.N_grid // 2
        
    def setup_agent_grid(self, X_POS_INIT, Y_POS_INIT, N_grid):
        DROP_SAND=False
        agent = Agent(x_pos_init=X_POS_INIT, y_pos_init=Y_POS_INIT)
        sandpile = Sandpile(N_grid=N_grid, agent=agent, DROP_SAND=DROP_SAND)
        return agent, sandpile

    def test_move_agent_left_from_middle(self,):
        EXPECTED_X_POS = self.X_POS_INIT - 1
        EXPECTED_Y_POS = self.Y_POS_INIT
        
        agent, sandpile = self.setup_agent_grid(self.X_POS_INIT, self.Y_POS_INIT, self.N_grid)

        sandpile.print_grid()
        print(agent.get_agent_pos())
        direction = Directions.LEFT

        sandpile.move_agent_in_direction(direction, agent)
        print(agent.get_agent_pos())
        self.assertEqual((EXPECTED_X_POS, EXPECTED_Y_POS), agent.get_agent_pos())
        self.assertEqual(agent.get_agent_pos(), sandpile.agent.get_agent_pos())
        self.assertTrue(sandpile.check_agent_is_in_grid(agent))

    def test_move_agent_right_from_middle(self,):
        EXPECTED_X_POS = self.X_POS_INIT + 1
        EXPECTED_Y_POS = self.Y_POS_INIT
        agent, sandpile = self.setup_agent_grid(self.X_POS_INIT, self.Y_POS_INIT, self.N_grid)

        sandpile.print_grid()
        print(agent.get_agent_pos())
        direction = Directions.RIGHT

        sandpile.move_agent_in_direction(direction, agent)
        print(agent.get_agent_pos())
        self.assertEqual((EXPECTED_X_POS, EXPECTED_Y_POS), agent.get_agent_pos())
        self.assertEqual(agent.get_agent_pos(), sandpile.agent.get_agent_pos())
        self.assertTrue(sandpile.check_agent_is_in_grid(agent))

    def test_move_agent_up_from_middle(self,):
        EXPECTED_X_POS = self.X_POS_INIT
        EXPECTED_Y_POS = self.Y_POS_INIT - 1
        agent, sandpile = self.setup_agent_grid(self.X_POS_INIT, self.Y_POS_INIT, self.N_grid)

        sandpile.print_grid()
        print(agent.get_agent_pos())
        direction = Directions.UP

        sandpile.move_agent_in_direction(direction, agent)
        print(agent.get_agent_pos())
        self.assertEqual((EXPECTED_X_POS, EXPECTED_Y_POS), agent.get_agent_pos())
        self.assertEqual(agent.get_agent_pos(), sandpile.agent.get_agent_pos())
        self.assertTrue(sandpile.check_agent_is_in_grid(agent))

    def test_move_agent_down_from_middle(self,):
        EXPECTED_X_POS = self.X_POS_INIT
        EXPECTED_Y_POS = self.Y_POS_INIT + 1
        agent, sandpile = self.setup_agent_grid(self.X_POS_INIT, self.Y_POS_INIT, self.N_grid)

        sandpile.print_grid()
        print(agent.get_agent_pos())
        direction = Directions.DOWN

        sandpile.move_agent_in_direction(direction, agent)
        print(agent.get_agent_pos())
        self.assertEqual((EXPECTED_X_POS, EXPECTED_Y_POS), agent.get_agent_pos())
        self.assertEqual(agent.get_agent_pos(), sandpile.agent.get_agent_pos())
        self.assertTrue(sandpile.check_agent_is_in_grid(agent))
    

class TestAgentMoveFuncsFromLeft(unittest.TestCase):

    def setUp(self):
        #middle position
        self.N_grid = 5 #number of cells per side
        self.X_POS_INIT = 0
        self.Y_POS_INIT = self.N_grid // 2
        
    def setup_agent_grid(self, X_POS_INIT, Y_POS_INIT, N_grid):
        DROP_SAND=False
        agent = Agent(x_pos_init=X_POS_INIT, y_pos_init=Y_POS_INIT)
        sandpile = Sandpile(N_grid=N_grid, agent=agent, DROP_SAND=DROP_SAND)
        return agent, sandpile

    def test_move_agent_left_from_left_bound(self,):
        EXPECTED_X_POS = self.X_POS_INIT - 1
        EXPECTED_Y_POS = self.Y_POS_INIT
        
        agent, sandpile = self.setup_agent_grid(self.X_POS_INIT, self.Y_POS_INIT, self.N_grid)

        sandpile.print_grid()
        print(agent.get_agent_pos())
        direction = Directions.LEFT

        sandpile.move_agent_in_direction(direction, agent)
        print(agent.get_agent_pos())
        self.assertEqual((EXPECTED_X_POS, EXPECTED_Y_POS), agent.get_agent_pos())
        self.assertEqual(agent.get_agent_pos(), sandpile.agent.get_agent_pos())
        self.assertFalse(sandpile.check_agent_is_in_grid(agent))
        self.assertFalse(sandpile.check_agent_is_in_grid(sandpile.agent))


class TestAgentMoveFuncsFromRight(unittest.TestCase):

    def setUp(self):
        #middle position
        self.N_grid = 5 #number of cells per side
        self.X_POS_INIT = self.N_grid - 1
        self.Y_POS_INIT = self.N_grid // 2
        
    def setup_agent_grid(self, X_POS_INIT, Y_POS_INIT, N_grid):
        DROP_SAND=False
        agent = Agent(x_pos_init=X_POS_INIT, y_pos_init=Y_POS_INIT)
        sandpile = Sandpile(N_grid=N_grid, agent=agent, DROP_SAND=DROP_SAND)
        return agent, sandpile

    def test_move_agent_right_from_right_bound(self,):
        EXPECTED_X_POS = self.X_POS_INIT + 1
        EXPECTED_Y_POS = self.Y_POS_INIT
        
        agent, sandpile = self.setup_agent_grid(self.X_POS_INIT, self.Y_POS_INIT, self.N_grid)

        sandpile.print_grid()
        print(agent.get_agent_pos())
        direction = Directions.RIGHT

        sandpile.move_agent_in_direction(direction, agent)
        print(agent.get_agent_pos())
        self.assertEqual((EXPECTED_X_POS, EXPECTED_Y_POS), agent.get_agent_pos())
        self.assertEqual(agent.get_agent_pos(), sandpile.agent.get_agent_pos())
        self.assertFalse(sandpile.check_agent_is_in_grid(agent))
        self.assertFalse(sandpile.check_agent_is_in_grid(sandpile.agent))


class TestAgentMoveFuncsFromTop(unittest.TestCase):

    def setUp(self):
        #middle position
        self.N_grid = 5 #number of cells per side
        self.X_POS_INIT = self.N_grid // 2
        self.Y_POS_INIT = 0
        
    def setup_agent_grid(self, X_POS_INIT, Y_POS_INIT, N_grid):
        DROP_SAND=False
        agent = Agent(x_pos_init=X_POS_INIT, y_pos_init=Y_POS_INIT)
        sandpile = Sandpile(N_grid=N_grid, agent=agent, DROP_SAND=DROP_SAND)
        return agent, sandpile

    def test_move_agent_up_from_top_bound(self,):
        EXPECTED_X_POS = self.X_POS_INIT
        EXPECTED_Y_POS = self.Y_POS_INIT - 1
        
        agent, sandpile = self.setup_agent_grid(self.X_POS_INIT, self.Y_POS_INIT, self.N_grid)

        sandpile.print_grid()
        print(agent.get_agent_pos())
        direction = Directions.UP

        sandpile.move_agent_in_direction(direction, agent)
        print(agent.get_agent_pos())
        self.assertEqual((EXPECTED_X_POS, EXPECTED_Y_POS), agent.get_agent_pos())
        self.assertEqual(agent.get_agent_pos(), sandpile.agent.get_agent_pos())
        self.assertFalse(sandpile.check_agent_is_in_grid(agent))
        self.assertFalse(sandpile.check_agent_is_in_grid(sandpile.agent))

class TestAgentMoveFuncsFromBot(unittest.TestCase):

    def setUp(self):
        #middle position
        self.N_grid = 5 #number of cells per side
        self.X_POS_INIT = self.N_grid // 2
        self.Y_POS_INIT = self.N_grid - 1
        
    def setup_agent_grid(self, X_POS_INIT, Y_POS_INIT, N_grid):
        DROP_SAND=False
        agent = Agent(x_pos_init=X_POS_INIT, y_pos_init=Y_POS_INIT)
        sandpile = Sandpile(N_grid=N_grid, agent=agent, DROP_SAND=DROP_SAND)
        return agent, sandpile

    def test_move_agent_down_from_bot_bound(self,):
        EXPECTED_X_POS = self.X_POS_INIT
        EXPECTED_Y_POS = self.Y_POS_INIT + 1
        
        agent, sandpile = self.setup_agent_grid(self.X_POS_INIT, self.Y_POS_INIT, self.N_grid)

        sandpile.print_grid()
        print(agent.get_agent_pos())
        direction = Directions.DOWN

        sandpile.move_agent_in_direction(direction, agent)
        print(agent.get_agent_pos())
        self.assertEqual((EXPECTED_X_POS, EXPECTED_Y_POS), agent.get_agent_pos())
        self.assertEqual(agent.get_agent_pos(), sandpile.agent.get_agent_pos())
        self.assertFalse(sandpile.check_agent_is_in_grid(agent))
        self.assertFalse(sandpile.check_agent_is_in_grid(sandpile.agent))

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestAgentMoveFuncsFromMiddle('test_move_agent_left_from_middle'))
    suite.addTest(TestAgentMoveFuncsFromMiddle('test_move_agent_right_from_middle'))
    suite.addTest(TestAgentMoveFuncsFromMiddle('test_move_agent_up_from_middle'))
    suite.addTest(TestAgentMoveFuncsFromMiddle('test_move_agent_down_from_middle'))
    suite.addTest(TestAgentMoveFuncsFromLeft('test_move_agent_left_from_left_bound'))
    suite.addTest(TestAgentMoveFuncsFromRight('test_move_agent_right_from_right_bound'))
    suite.addTest(TestAgentMoveFuncsFromTop('test_move_agent_up_from_top_bound'))
    suite.addTest(TestAgentMoveFuncsFromBot('test_move_agent_down_from_bot_bound'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())