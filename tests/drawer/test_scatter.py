import pytest

import numpy as np

from mathtools.drawer import Scatter


class Test_Drawer_Scatter:
    def test_normal_scatter(self):
        x = np.linspace(0, 15, 100)
        y = np.sin(x)
        sc = Scatter("Test", "X", "Y")
        sc.scatter({
            "Data_1": {
                "x_data": x,
                "y_data": y
            }
        })
        sc.scatter({
            "Data_2": {
                "y_data": y
            }
        })

    def test_kmeans_cluster(self):
        x, y = np.random.rand(100, 2), np.random.randint(0, 3, 100)
        sc = Scatter("KMeans01", "X", "Y")
        sc.cluster({
            "Data_1": {
                "x_data": x[:, 0],
                "y_data": x[:, 1],
                "sizes": 50,
                "category_result": y,
                "category_names": ["A", "B", "C"]
            }
        }).with_centroid(["A", "B"])

    def test_mixed_scatter(self):
        x = np.linspace(0, 15, 100)
        y = np.sin(x)
        sc = Scatter("Test", "X", "Y")
        sc.scatter({
            "Data_1": {
                "x_data": x,
                "y_data": y
            }
        })
        x, y = np.random.rand(100, 2), np.random.randint(0, 3, 100)
        sc.cluster({
            "Data_2": {
                "x_data": x[:, 0],
                "y_data": x[:, 1],
                "sizes": 50,
                "category_result": y,
                "category_names": ["A", "B", "C"]
            }
        }).with_centroid(["A", "B"])

        with pytest.warns(UserWarning):
            sc.with_centroid(["Data_1"])
