import pytest

import numpy as np

from mathtools.db import DataHandler


class TestDataHandler:
    def test_save_numpy(self, tmpdir):
        test_array = np.random.rand(10, 10)
        dh = DataHandler(str(tmpdir / "test_db"))
        dh = DataHandler(str(tmpdir / "test_db.db"))
        dh.save_numpy("test.test_npy1.1", test_array)
        with pytest.warns(UserWarning):
            dh.save_numpy("test.test_npy1.1", test_array)
        assert np.array_equal(test_array, dh.load_numpy("test.test_npy1.1"))

        test_array2 = np.random.rand(10, 10)
        dh.save_numpy("test.test_npy1.1", test_array2, is_force=True)
        assert np.array_equal(test_array2, dh.load_numpy("test.test_npy1.1"))

        with pytest.raises(KeyError) as e:
            dh.load_numpy("test.test_npy2")
        assert "'The path test.test_npy2 is not exists.'" == str(e.value)

        with pytest.raises(KeyError) as e:
            dh.save_numpy("", test_array)

        with pytest.raises(KeyError) as e:
            dh.load_numpy("1.2.3")

        with pytest.raises(KeyError) as e:
            dh.load_numpy("")

    def test_save_object(self, tmpdir):
        test_obj = {"test": "test", "test2": 1, "test3": [1, 2, 3], "test4": {"test": "test"}}
        dh = DataHandler(str(tmpdir / "test_db"))
        dh.save_obj("test.test_obj1", test_obj)
        with pytest.warns(UserWarning):
            dh.save_obj("test.test_obj1", test_obj)
        assert test_obj == dh.load_obj("test.test_obj1")

        test_obj2 = {"test": "test2", "test2": 2, "test3": [4, 5, 6], "test4": {"test": "test2"}}
        dh.save_obj("test.test_obj1", test_obj2, is_force=True)
        assert test_obj2 == dh.load_obj("test.test_obj1")

        with pytest.raises(KeyError) as e:
            dh.load_numpy("test.test_npy2")
        assert "'The path test.test_npy2 is not exists.'" == str(e.value)

        with pytest.raises(KeyError) as e:
            dh.save_obj("", test_obj)

        with pytest.raises(KeyError) as e:
            dh.load_obj("1.2.3")

        with pytest.raises(KeyError) as e:
            dh.load_obj("")
