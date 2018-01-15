def test():
    """ Stub for running all unit test utilities """
    import unittest
    from test_utils import TestUtils

    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestUtils))
    unittest.TextTestRunner().run(test_suite)