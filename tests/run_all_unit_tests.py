import unittest
import test_agent_move_funcs
import test_maxAgent_move_funcs
import test_sandpile_mechanics


if __name__ == '__main__':

    agent_move_suite = test_agent_move_funcs.suite()
    max_agent_move_suite = test_maxAgent_move_funcs.suite()
    sandpile_suite = test_sandpile_mechanics.suite()

    all_suites_list = [agent_move_suite, max_agent_move_suite, sandpile_suite]
    all_suites = unittest.TestSuite(all_suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(all_suites)
    print(results)