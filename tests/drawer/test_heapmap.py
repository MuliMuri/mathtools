import pytest

import numpy as np

from mathtools.drawer import Heatmap


class Test_Drawer_Heatmap:
    def test_normal_imshow(self):
        test_heat_data = np.random.rand(20, 21)

        heatmap = Heatmap("Test Heatmap", "X Label", "Y Label")
        heatmap.imshow({
            "Test Heatmap": {
                "heat_data": test_heat_data,
                "x_ticks": np.arange(21),
                "y_ticks": np.arange(20)
            }
        })

    def test_normal_contourf(self):
        test_heat_data = np.random.rand(20, 21)

        heatmap = Heatmap("Test Heatmap", "X Label", "Y Label")
        heatmap.contourf({
            "Test Heatmap": {
                "heat_data": test_heat_data,
                "x_ticks": np.arange(21),
                "y_ticks": np.arange(20)
            }
        })

    def test_full_imshow(self):
        test_heat_data = np.random.rand(20, 21)

        heatmap = Heatmap("Test Heatmap", "X Label", "Y Label")
        heatmap.imshow({
            "Test Heatmap": {
                "heat_data": test_heat_data,
                "x_ticks": np.arange(21),
                "y_ticks": np.arange(20)
            }
        }, 90).with_colorbar().with_specific_text()

    def test_full_contourf(self):
        test_heat_data = np.random.rand(20, 21)

        heatmap = Heatmap("Test Heatmap", "X Label", "Y Label")
        heatmap.contourf({
            "Test Heatmap": {
                "heat_data": test_heat_data,
                "x_ticks": np.arange(21),
                "y_ticks": np.arange(20)
            }
        }, 90).with_colorbar().with_specific_text()

    def test_unmatch_ticks(self):
        test_heat_data = np.random.rand(20, 21)

        heatmap = Heatmap("Test Heatmap", "X Label", "Y Label")
        with pytest.raises(ValueError) as e:
            heatmap.imshow({
                "Test Heatmap": {
                    "heat_data": test_heat_data,
                    "x_ticks": np.arange(1),
                    "y_ticks": np.arange(10)
                }
            })
        assert "The length of x_ticks and y_ticks must match the shape of `heat_data`" == str(e.value)

        with pytest.raises(ValueError) as e:
            heatmap.contourf({
                "Test Heatmap": {
                    "heat_data": test_heat_data,
                    "x_ticks": np.arange(1),
                    "y_ticks": np.arange(10)
                }
            })
        assert "The length of x_ticks and y_ticks must match the shape of `heat_data`" == str(e.value)

    def test_multiple_heatmaps(self):
        test_heat_data = np.random.rand(20, 21)

        heatmap = Heatmap("Test Heatmap", "X Label", "Y Label")
        heatmap.imshow({
            "Test Heatmap": {
                "heat_data": test_heat_data,
                "x_ticks": np.arange(21),
                "y_ticks": np.arange(20)
            }
        })

        with pytest.warns(UserWarning):
            heatmap.imshow({
                "Test Heatmap": {
                    "heat_data": test_heat_data,
                    "x_ticks": np.arange(21),
                    "y_ticks": np.arange(20)
                }
            })

        with pytest.warns(UserWarning):
            heatmap.contourf({
                "Test Heatmap": {
                    "heat_data": test_heat_data,
                    "x_ticks": np.arange(21),
                    "y_ticks": np.arange(20)
                }
            })

    def test_multiple_data_on_same_plot(self):
        test_heat_data = np.random.rand(20, 21)

        heatmap = Heatmap("Test Heatmap", "X Label", "Y Label")

        with pytest.raises(ValueError) as e:
            heatmap.imshow({
                "Test Heatmap2": {
                    "heat_data": test_heat_data,
                    "x_ticks": np.arange(21),
                    "y_ticks": np.arange(20)
                },
                "Test Heatmap": {
                    "heat_data": test_heat_data,
                    "x_ticks": np.arange(21),
                    "y_ticks": np.arange(20)
                }
            })
            assert "Only one heatmap can be plotted at a time" == str(e.value)

        with pytest.raises(ValueError) as e:
            heatmap.contourf({
                "Test Heatmap2": {
                    "heat_data": test_heat_data,
                    "x_ticks": np.arange(21),
                    "y_ticks": np.arange(20)
                },
                "Test Heatmap": {
                    "heat_data": test_heat_data,
                    "x_ticks": np.arange(21),
                    "y_ticks": np.arange(20)
                }
            })
            assert "Only one heatmap can be plotted at a time" == str(e.value)
