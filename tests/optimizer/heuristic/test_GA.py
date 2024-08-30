import numpy as np

from mathtools.optimizer.heuristic import GA


class TestOptimizerHeuristicGA:
    def test_func1(self):
        def func(x):
            x1 = x[0]
            x2 = x[1]

            res = 4 * x1 ** 2 - 2.1 * x1 ** 4 + x1 ** 6 / 3 + x1 * x2 - 4 * x2 ** 2 + 4 * x2 ** 4
            return res

        def st_func(x):
            x1 = x[0]
            x2 = x[1]

            if -1 <= x1 <= 1 and -1 <= x2 <= 1:
                return True
            else:
                return False

        ga = GA(func, 2, st_func=st_func, dna_size=24, pop_size=200, generation=200, bounds=np.array([[-10, 10], [-10, 10]]))

        ga.run()
