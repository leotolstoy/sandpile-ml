import numpy as np
import os, sys
sys.path.append("../")
from util import Directions
from sandpile import Sandpile
from agents import Agent
import unittest

class TestSandpileMechanics(unittest.TestCase):

    def setUp(self):
        #middle position
        self.N_grid = 5 #number of cells per side
        
    def setup_sandpile_grid_no_agent(self, N_grid, max_grains=4):
        DROP_SAND=False
        sandpile = Sandpile(N_grid=N_grid, DROP_SAND=DROP_SAND, MAXIMUM_GRAINS=max_grains)
        return sandpile

    def test_sandgrain_drop_at_location(self,):
        
        X_POS = self.N_grid//2
        Y_POS = self.N_grid//2
        
        sandpile = self.setup_sandpile_grid_no_agent(self.N_grid)

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
        sandpile = self.setup_sandpile_grid_no_agent(self.N_grid, max_grains=max_grains)

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
        sandpile = self.setup_sandpile_grid_no_agent(self.N_grid, max_grains=max_grains)

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

        sandpile.step()

        print(sandpile.avalanche_sizes)

        self.assertEqual(EXPECTED_AVALANCHE_SIZE, sandpile.avalanche_sizes[0])

    def test_avalanche_at_side_loses_grains(self,):

        # set up sandpile at avalanche conditions
        max_grains = 4
        sandpile = self.setup_sandpile_grid_no_agent(self.N_grid, max_grains=max_grains)

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
        sandpile = self.setup_sandpile_grid_no_agent(self.N_grid, max_grains=max_grains)

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

    def test_two_avalanches_recorded_properly(self,):

        # set up sandpile at avalanche conditions
        max_grains = 4
        sandpile = self.setup_sandpile_grid_no_agent(self.N_grid, max_grains=max_grains)

        # define sandpile scenario
        # [[0. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]
        # [0. 0. 4. 3. 0.]
        # [0. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]]

        sandpile.grid[2,2] = max_grains
        sandpile.grid[2,3] = max_grains - 1
        print('sandpile before')
        sandpile.print_grid()

        # define expected grid
        # [[0. 0. 0. 0. 0.]
        # [0. 0. 1. 1. 0.]
        # [0. 1. 1. 0. 1.]
        # [0. 0. 1. 1. 0.]
        # [0. 0. 0. 0. 0.]]

        EXPECTED_GRID = np.zeros((self.N_grid,self.N_grid))
        EXPECTED_GRID[1,2] = 1
        EXPECTED_GRID[1,3] = 1
        EXPECTED_GRID[2,1] = 1
        EXPECTED_GRID[2,2] = 1
        EXPECTED_GRID[2,4] = 1
        EXPECTED_GRID[3,2] = 1
        EXPECTED_GRID[3,3] = 1
        EXPECTED_AVALANCHE_SIZE = 2

        print('expected grid')
        print(EXPECTED_GRID)

        #step which should avalanche twice
        sandpile.step()

        arraysEqual = np.array_equal(EXPECTED_GRID, sandpile.get_sandpile())
        sandpile.print_grid()
        self.assertTrue(arraysEqual)

        self.assertEqual(EXPECTED_AVALANCHE_SIZE, sandpile.avalanche_sizes[0])

    def test_avalanche_moves_agent(self,):
        
        # define sandpile scenario
        # [[0. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]
        # [0. 0. 4. 0. 0.]
        # [0. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]]
        X_POS = 2
        Y_POS = 2

        # set up sandpile at avalanche conditions
        max_grains = 4
        DROP_SAND=False
        agent = Agent(x_pos_init=X_POS, y_pos_init=Y_POS)
        sandpile = Sandpile(N_grid=self.N_grid, DROP_SAND=DROP_SAND, agents=[agent], MAXIMUM_GRAINS=max_grains)

        
        sandpile.grid[Y_POS, X_POS] = max_grains
        print('sandpile before:')
        print(agent.get_agent_pos())
        sandpile.print_grid_and_agent_pos(agent)


        #step, which should avalanche
        sandpile.step()
        
        print('sandpile after:')
        print(agent.get_agent_pos())
        sandpile.print_grid_and_agent_pos(agent)


        self.assertNotEqual(agent.get_agent_pos(), (Y_POS, X_POS))

    def test_initial_grid_correctly_used(self,):
        # define initial grid
        initial_grid = np.random.rand(self.N_grid, self.N_grid)
        sandpile = Sandpile(N_grid=self.N_grid, initial_grid=initial_grid)

        arraysEqual = np.array_equal(initial_grid, sandpile.get_sandpile())
        self.assertTrue(arraysEqual)

    def test_initial_grid_not_square_raises_exception(self,):
        # define initial grid that is not square
        initial_grid = np.random.rand(self.N_grid, self.N_grid+1)

        with self.assertRaises(AssertionError) as cm:
            sandpile=Sandpile(N_grid=self.N_grid, initial_grid=initial_grid)

        e = cm.exception
        self.assertEqual(AssertionError, e.__class__)

    def test_drop_sandgrain_preset_location(self,):

        max_grains = 4
        DROP_SAND=False

        # define expected grid
        # [[1. 1. 0. 0. 1.]
        # [0. 1. 0. 0. 0.]
        # [0. 0. 1. 0. 0.]
        # [0. 0. 0. 0. 0.]
        # [0. 0. 0. 0. 0.]]

        # set up preset sandgrain locations (x,y) / (j,i)
        sandgrain_locs = np.array([[0,0],
                                   [1,1],
                                   [1,0],
                                   [self.N_grid//2, self.N_grid//2],
                                   [self.N_grid-1, 0]])
        
        sandpile = Sandpile(N_grid=self.N_grid, DROP_SAND=True, MAXIMUM_GRAINS=max_grains, grain_loc_order=sandgrain_locs)
        
        EXPECTED_GRID = np.zeros((self.N_grid,self.N_grid))

        N = sandgrain_locs.shape[0]
        for i in range(N):
            loc = sandgrain_locs[i,:]
            # print(loc)
            EXPECTED_GRID[loc[1], loc[0]] = 1

        sandpile.print_grid()

        print('expected grid')
        print(EXPECTED_GRID)

        #step
        for i in range(N):
            sandpile.step()

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
    suite.addTest(TestSandpileMechanics('test_two_avalanches_recorded_properly'))
    suite.addTest(TestSandpileMechanics('test_avalanche_moves_agent'))   
    suite.addTest(TestSandpileMechanics('test_initial_grid_correctly_used'))
    suite.addTest(TestSandpileMechanics('test_initial_grid_not_square_raises_exception'))
    suite.addTest(TestSandpileMechanics('test_drop_sandgrain_preset_location'))
    
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    results = runner.run(suite())
    print(results)