import numpy as np

from mathtools.optimizer.heuristic.SA import SA
from mathtools.algorithm.perturbation import Perturbation


class TestOptimizerHeuristicSA:
    def test_func1(self):
        def func(x):
            x1 = x[0]
            x2 = x[1]

            res = 4 * x1 ** 2 - 2.1 * x1 ** 4 + x1 ** 6 / 3 + x1 * x2 - 4 * x2 ** 2 + 4 * x2 ** 4
            return res

        def st_func(x):
            x1 = x[0]
            x2 = x[1]

            if -5 <= x1 <= 5 and -5 <= x2 <= 5:
                return True
            else:
                return False

        sa = SA(init_x=np.array([0, 0]),
                target_func=func,
                st_func=st_func,
                perturbation_func=Perturbation.gaussian,
                Tf=1e-5,
                is_log=True,
                iter=100)

        result = sa.run()

        assert result[1] - -1.031628 < 1e-3

    def test_other_perturbation_funcs(self):
        def func(x):
            x1 = x[0]
            x2 = x[1]

            res = 4 * x1 ** 2 - 2.1 * x1 ** 4 + x1 ** 6 / 3 + x1 * x2 - 4 * x2 ** 2 + 4 * x2 ** 4
            return res

        def st_func(x):
            x1 = x[0]
            x2 = x[1]

            if -5 <= x1 <= 5 and -5 <= x2 <= 5:
                return True
            else:
                return False

        SA(init_x=np.array([0, 0]),
           target_func=func,
           st_func=st_func,
           perturbation_func=Perturbation.gaussian,
           Tf=1e-5,
           is_log=True,
           iter=100).run()

        SA(init_x=np.array([0, 0]),
           target_func=func,
           st_func=st_func,
           perturbation_func=Perturbation.uniform_random,
           Tf=1e-5,
           is_log=True,
           iter=100).run()

        SA(init_x=np.array([0, 0]),
           target_func=func,
           st_func=st_func,
           perturbation_func=Perturbation.neighbor,
           Tf=1e-5,
           is_log=True,
           iter=100).run()

        SA(init_x=np.array([0, 0]),
           target_func=func,
           st_func=st_func,
           perturbation_func=Perturbation.cauchy,
           Tf=1e-5,
           is_log=True,
           iter=100).run()
