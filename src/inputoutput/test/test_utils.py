from unittest import TestCase

from unipath import Path

from inputoutput.utils import build_result_path, build_xs_path


class TestBuildResultPath(TestCase):

    def setUp(self):
        super(TestBuildResultPath, self).setUp()
        self._directory = Path('/Users/laura/Repositories/stage/data/simulated/small')
        self._data_set_file = Path('/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90.txt')
        self._estimator = 'sambe'

    def test_no_sensitivity(self):
        actual = build_result_path(self._directory, self._data_set_file, self._estimator)
        expected = Path('/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90_sambe.txt')
        self.assertEqual(actual, expected)

    def test_with_sensitivity(self):
        sensitivity = 'silverman'
        actual = build_result_path(self._directory, self._data_set_file, self._estimator, sensitivity)
        expected = Path('/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90_sambe_silverman.txt')
        self.assertEqual(actual, expected)

    def test_with_grid(self):
        sensitivity = 'silverman'
        grid = 'grid256'
        actual = build_result_path(self._directory, self._data_set_file, self._estimator, sensitivity, grid)
        expected = Path(
            '/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90_sambe_silverman_grid256.txt'
        )
        self.assertEqual(actual, expected)

    def test_with_empty_list(self):
        actual = build_result_path(self._directory, self._data_set_file, self._estimator, '')
        expected = Path('/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90_sambe.txt')
        self.assertEqual(actual, expected)


class TestBuildXSPath(TestCase):
    def setUp(self):
        super(TestBuildXSPath, self).setUp()
        self._directory = Path('/Users/laura/Repositories/stage/data/simulated/small')
        self._data_set_file = Path('/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90.txt')

    def test_no_args(self):
        actual = build_xs_path(self._directory, self._data_set_file)
        expected = Path(
            '/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90.txt'
        )
        self.assertEqual(actual, expected)

    def test_with_args(self):
        grid_string = 'grid_128'
        actual = build_xs_path(self._directory, self._data_set_file, grid_string)
        expected = Path(
            '/Users/laura/Repositories/stage/data/simulated/small/ferdosi_1_90_grid_128.txt'
        )
        self.assertEqual(actual, expected)
