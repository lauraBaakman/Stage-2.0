import unittest
import warnings

import kde.tests.test_sambe as test_sambe

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    loader = unittest.TestLoader()

    all_tests = unittest.TestSuite(loader.discover('.'))

    temp_suite = unittest.TestSuite()
    temp_suite.addTest(test_sambe.Test_ShapeAdaptiveMBE_Python(
        'test_xis_is_not_xs')
    )
    temp_suite.addTest(test_sambe.Test_ShapeAdaptiveMBE_C(
        'test_xis_is_not_xs')
    )

    # kde_tests = unittest.TestSuite(loader.discover('./kde/tests'))
    # kernel_tests = unittest.TestSuite(loader.discover('./kde/kernels/tests'))
    # util_tests = unittest.TestSuite(loader.discover('./kde/utils/tests'))

    unittest.TextTestRunner(verbosity=2).run(temp_suite)
