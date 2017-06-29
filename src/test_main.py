import unittest
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    loader = unittest.TestLoader()

    all_tests = unittest.TestSuite(loader.discover('.'))

    kde_tests = unittest.TestSuite(loader.discover('./kde/tests'))
    kernel_tests = unittest.TestSuite(loader.discover('./kde/kernels/tests'))
    util_tests = unittest.TestSuite(loader.discover('./kde/utils/tests'))

    unittest.TextTestRunner(verbosity=1).run(all_tests)
