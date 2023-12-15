import unittest
import test_agent_move_funcs
import test_maxAgent_move_funcs
import test_specificValueAgent_move_funcs
import test_sandpile_mechanics
import test_util_funcs


if __name__ == '__main__':

    agent_move_suite = test_agent_move_funcs.suite()
    max_agent_move_suite = test_maxAgent_move_funcs.suite()
    ssv_agent_move_suite = test_specificValueAgent_move_funcs.suite()
    sandpile_suite = test_sandpile_mechanics.suite()
    test_util_suite = test_util_funcs.suite()

    all_suites_list = [agent_move_suite, max_agent_move_suite, ssv_agent_move_suite, sandpile_suite, test_util_suite]
    all_suites = unittest.TestSuite(all_suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(all_suites)
    print(results)