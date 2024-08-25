import numpy as np

from mathtools.drawer import Plot


class Test_Drawer_Heatmap:
    def test_normal_plot(self):
        test_x = np.arange(0, 100, 1)
        test_y = np.random.rand(100)

        plot_noX = Plot("Test Plot", "X Label", "Y Label")
        plot_noX.plot({
            "Test Plot": {
                "y_data": test_y
            }
        })

        plot_X = Plot("Test Plot", "X Label", "Y Label")
        plot_X.plot({
            "Test Plot": {
                "x_data": test_x,
                "y_data": test_y
            }
        })

    def test_normal_spline(self):
        test_x = np.arange(0, 100, 1)
        test_y = np.random.rand(100)

        plot_noX = Plot("Test Plot", "X Label", "Y Label")
        plot_noX.spline({
            "Test Plot": {
                "y_data": test_y
            }
        })

        plot_X = Plot("Test Plot", "X Label", "Y Label")
        plot_X.spline({
            "Test Plot": {
                "x_data": test_x,
                "y_data": test_y
            }
        })

    def test_full_plot(self):
        test_x = np.arange(0, 100, 1)
        test_y1 = np.random.rand(100)
        test_y2 = np.random.rand(100)

        plot = Plot("Test Plot", "X Label", "Y Label")

        plot.plot({
            "p1": {
                "y_data": test_y1,
                "x_data": test_x
            }
        }).spline({
            "p2": {
                "y_data": test_y2
            }
        }).with_error_std(['p1']).with_local_zoom(['p2'], is_zoom_ylims=True).with_density(['p1', 'p2'])
