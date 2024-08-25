import pytest

import numpy as np

from mathtools.drawer import Bar


class Test_Drawer_Bar:
    def test_normal_bar(self):
        test_bar_data = np.random.randn((5))

        bar = Bar("Test Bar", "X Label", "Y Label")
        bar.bar({
            "Test Bar": {
                "y_data": test_bar_data,
                "x_labels": np.arange(5)
            }
        })

    def test_normal_group_bar(self):
        test_bar_data = np.random.randn(5, 5)

        bar = Bar("Test Bar", "X Label", "Y Label")
        bar.group_bar({
            "Data_1": {
                "y_data": test_bar_data[0],
                "x_labels": np.arange(5)
            },
            "Data_2": {
                "y_data": test_bar_data[1],
                "x_labels": np.arange(5)
            },
            "Data_3": {
                "y_data": test_bar_data[2],
                "x_labels": np.arange(5)
            },
            "Data_4": {
                "y_data": test_bar_data[3],
                "x_labels": np.arange(5)
            },
            "Data_5": {
                "y_data": test_bar_data[4],
                "x_labels": np.arange(5)
            }
        })

    def test_full_bar(self):
        test_bar_data = np.random.randn(5)

        bar = Bar("Test Bar", "X Label", "Y Label")
        bar.bar({
            "Test Bar": {
                "y_data": test_bar_data,
                "x_labels": np.arange(5)
            }
        }).with_top_label(['Test Bar'])

    def test_full_group_bar(self):
        test_bar_data = np.random.randn(5, 5)

        bar = Bar("Test Bar", "X Label", "Y Label")
        bar.group_bar({
            "Data_1": {
                "y_data": test_bar_data[0],
                "x_labels": np.arange(5)
            },
            "Data_2": {
                "y_data": test_bar_data[1],
                "x_labels": np.arange(5)
            },
            "Data_3": {
                "y_data": test_bar_data[2],
                "x_labels": np.arange(5)
            },
            "Data_4": {
                "y_data": test_bar_data[3],
                "x_labels": np.arange(5)
            },
            "Data_5": {
                "y_data": test_bar_data[4],
                "x_labels": np.arange(5)
            }
        }).with_top_label(['Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5'])

    def test_empty_bar(self):
        bar = Bar("Test Bar", "X Label", "Y Label")

        with pytest.warns(UserWarning):
            bar.with_top_label(['Test Bar'])

    def test_multiple_bars(self):
        test_bar_data = np.random.randn(5)
        test_group_bar_data = np.random.rand(5, 5)

        bar = Bar("Test Bar", "X Label", "Y Label")
        bar.bar({
            "Test Bar": {
                "y_data": test_bar_data,
                "x_labels": np.arange(5)
            }
        })

        with pytest.warns(UserWarning):
            bar.bar({
                "Test Bar": {
                    "y_data": test_bar_data,
                    "x_labels": np.arange(5)
                }
            })

        with pytest.warns(UserWarning):
            bar.group_bar({
                "Data_1": {
                    "y_data": test_group_bar_data[0],
                    "x_labels": np.arange(5)
                },
                "Data_2": {
                    "y_data": test_group_bar_data[1],
                    "x_labels": np.arange(5)
                },
                "Data_3": {
                    "y_data": test_group_bar_data[2],
                    "x_labels": np.arange(5)
                },
                "Data_4": {
                    "y_data": test_group_bar_data[3],
                    "x_labels": np.arange(5)
                },
                "Data_5": {
                    "y_data": test_group_bar_data[4],
                    "x_labels": np.arange(5)
                }
            })
