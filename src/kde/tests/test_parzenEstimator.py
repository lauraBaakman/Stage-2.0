from unittest import TestCase

from kde.parzen import _ParzenEstimator_C, _ParzenEstimator_Python


class TestParzenEstimator(TestCase):
    def test_estimate_python(self):
        self.fail()

    def test_estimate_C(self):
        self.fail()


class ParzenEstimatorImpAbstractTest(object):
    def setUp(self):
        super().setUp()
        self._estimator_class = None

    def test_estimate(self):
        self.fail()


class Test_ParzenEstimator_Python(ParzenEstimatorImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._estimator_class = _ParzenEstimator_Python


class Test_ParzenEstimator_C(ParzenEstimatorImpAbstractTest, TestCase):
    def setUp(self):
        super().setUp()
        self._estimator_class = _ParzenEstimator_C
