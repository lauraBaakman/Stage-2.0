import unittest
import warnings

import kde.kernels.tests.test_shapeAdaptiveEpanechnikov as test_saEpanechnikov

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    loader = unittest.TestLoader()

    all_tests = unittest.TestSuite(loader.discover('.'))

    temp_suite = unittest.TestSuite()
    temp_suite.addTest(test_saEpanechnikov.TestShapeAdaptiveEpanechnikov(
        'test_default_implementation_single_pattern_l_eq_1')
    )

    # kde_tests = unittest.TestSuite(loader.discover('./kde/tests'))
    # kernel_tests = unittest.TestSuite(loader.discover('./kde/kernels/tests'))
    # util_tests = unittest.TestSuite(loader.discover('./kde/utils/tests'))

    unittest.TextTestRunner(verbosity=2).run(temp_suite)
