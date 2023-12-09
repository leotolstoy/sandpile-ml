import numpy as np
import os, sys
sys.path.append("../")
from util import Directions
from sandpile import Sandpile
from agents import MaxAgent
import unittest

class TestSandpileMechanics(unittest.TestCase):

    def setUp(self):
        #middle position
        self.N_grid = 5 #number of cells per side
        
    def setup_sandpile_grid(self, N_grid, max_grains=4):
        DROP_SAND=False
        sandpile = Sandpile(N_grid=N_grid, DROP_SAND=DROP_SAND, MAXIMUM_GRAINS=max_grains)
        return sandpile

    def test_sandgrain_drop_at_location(self,):
        
        X_POS = self.N_grid//2
        Y_POS = self.N_grid//2
        
        sandpile = self.setup_sandpile_grid(self.N_grid)

        sandpile.print_grid()

        #drop a sandgrain at location
        sandpile.drop_sandgrain_at_pos(X_POS, Y_POS)

        # define expected grid
        EXPECTED_GRID = np.zeros((self.N_grid,self.N_grid))
        EXPECTED_GRID[Y_POS, X_POS] = 1
        print(EXPECTED_GRID)

        arraysEqual = np.array_equal(EXPECTED_GRID, sandpile.get_sandpile())
        sandpile.print_grid()
        self.assertTrue(arraysEqual)

    def test_drop_on_maxgrains_causes_avalanche(self,):

        # set up sandpile at avalanche conditions
        max_grains = 4
        sandpile = self.setup_sandpile_grid(self.N_grid, max_grains=max_grains)

        # define sandpile scenario
        # [[0. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]
        # [0. 0. 3. 0. 0.]
        # [0. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]]
        DROP_X_POS = self.N_grid//2
        DROP_Y_POS = self.N_grid//2

        sandpile.grid[DROP_Y_POS, DROP_X_POS] = max_grains - 1
        print('sandpile before')
        sandpile.print_grid()

        # define expected grid
        # [[0. 0. 0. 0. 0.]
        # [0. 0. 1. 0. 0.]
        # [0. 1. 0. 1. 0.]
        # [0. 0. 1. 0. 0.]
        # [0. 0. 0. 0. 0.]]

        EXPECTED_GRID = np.zeros((self.N_grid,self.N_grid))
        EXPECTED_GRID[DROP_Y_POS, DROP_X_POS-1] = 1
        EXPECTED_GRID[DROP_Y_POS, DROP_X_POS+1] = 1
        EXPECTED_GRID[DROP_Y_POS-1, DROP_X_POS] = 1
        EXPECTED_GRID[DROP_Y_POS+1, DROP_X_POS] = 1
        print('expected grid')
        print(EXPECTED_GRID)

        #drop a sandgrain at location
        sandpile.drop_sandgrain_at_pos(DROP_X_POS, DROP_Y_POS)

        sandpile.step()

        arraysEqual = np.array_equal(EXPECTED_GRID, sandpile.get_sandpile())
        sandpile.print_grid()
        self.assertTrue(arraysEqual)

    def test_avalanche_size_recorded_properly(self,):

        # set up sandpile at avalanche conditions
        max_grains = 4
        sandpile = self.setup_sandpile_grid(self.N_grid, max_grains=max_grains)

        # define sandpile scenario
        # [[0. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]
        # [0. 0. 4. 0. 0.]
        # [0. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]]
        DROP_X_POS = self.N_grid//2
        DROP_Y_POS = self.N_grid//2

        sandpile.grid[DROP_Y_POS, DROP_X_POS] = max_grains
        print('sandpile before')
        sandpile.print_grid()

        EXPECTED_AVALANCHE_SIZE = 1

        #step, which should avalanche
        sandpile.step()

        print(sandpile.avalanche_sizes)
        self.assertTrue(sandpile.is_avalanching)

        #step again, which should stop avalanching
        print('NEXT STEP')
        sandpile.step()
        self.assertFalse(sandpile.is_avalanching)

        self.assertEqual(EXPECTED_AVALANCHE_SIZE, sandpile.avalanche_sizes[0])

    def test_avalanche_at_side_loses_grains(self,):

        # set up sandpile at avalanche conditions
        max_grains = 4
        sandpile = self.setup_sandpile_grid(self.N_grid, max_grains=max_grains)

        # define sandpile scenario, at left edge
        # [[0. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]
        # [4. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]]
        DROP_X_POS = 0
        DROP_Y_POS = self.N_grid//2

        sandpile.grid[DROP_Y_POS, DROP_X_POS] = max_grains
        print('sandpile before')
        sandpile.print_grid()

        #step, which should avalanche
        sandpile.step()

        print(sandpile.avalanche_sizes)
        self.assertTrue(sandpile.is_avalanching)

        # define expected grid
        # [[0. 0. 0. 0. 0.]
        # [1. 0. 0. 0. 0.]
        # [0. 1. 0. 0. 0.]
        # [1. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]]

        EXPECTED_GRID = np.zeros((self.N_grid,self.N_grid))
        EXPECTED_GRID[DROP_Y_POS, DROP_X_POS+1] = 1
        EXPECTED_GRID[DROP_Y_POS-1, DROP_X_POS] = 1
        EXPECTED_GRID[DROP_Y_POS+1, DROP_X_POS] = 1
        print('expected grid')
        print(EXPECTED_GRID)

        arraysEqual = np.array_equal(EXPECTED_GRID, sandpile.get_sandpile())
        sandpile.print_grid()
        self.assertTrue(arraysEqual)

    def test_avalanche_at_corner_loses_grains(self,):

        # set up sandpile at avalanche conditions
        max_grains = 4
        sandpile = self.setup_sandpile_grid(self.N_grid, max_grains=max_grains)

        # define sandpile scenario, at top left corner
        # [[4. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]]
        DROP_X_POS = 0
        DROP_Y_POS = 0

        sandpile.grid[DROP_Y_POS, DROP_X_POS] = max_grains
        print('sandpile before')
        sandpile.print_grid()

        #step, which should avalanche
        sandpile.step()

        print(sandpile.avalanche_sizes)
        self.assertTrue(sandpile.is_avalanching)

        # define expected grid
        # [[0. 1. 0. 0. 0.]
        # [1. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]]

        EXPECTED_GRID = np.zeros((self.N_grid,self.N_grid))
        EXPECTED_GRID[DROP_Y_POS, DROP_X_POS+1] = 1
        EXPECTED_GRID[DROP_Y_POS+1, DROP_X_POS] = 1
        print('expected grid')
        print(EXPECTED_GRID)

        arraysEqual = np.array_equal(EXPECTED_GRID, sandpile.get_sandpile())
        sandpile.print_grid()
        self.assertTrue(arraysEqual)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestSandpileMechanics('test_sandgrain_drop_at_location'))
    suite.addTest(TestSandpileMechanics('test_drop_on_maxgrains_causes_avalanche'))
    suite.addTest(TestSandpileMechanics('test_avalanche_size_recorded_properly'))
    suite.addTest(TestSandpileMechanics('test_avalanche_at_side_loses_grains'))
    suite.addTest(TestSandpileMechanics('test_avalanche_at_corner_loses_grains'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())