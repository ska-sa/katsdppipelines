import unittest

from katacomb.util import parse_python_assigns

class TestUtils(unittest.TestCase):

    def test_parse_python_assigns(self):
        """ Test basic assignments """

        assign_str = ("scans='track';"
                      "spw=0;"
                      "pol='HH,VV';"
                      "targets='PHOENIX_DEEP';"
                      "channels=slice(0,4096)")

        D = parse_python_assigns(assign_str)

        self.assertTrue(D == {
            "scans": "track",
            "spw": 0,
            "pol": "HH,VV",
            "targets": "PHOENIX_DEEP",
            "channels": slice(0, 4096)})

        # Test failure on illegal function
        with self.assertRaises(ValueError) as cm:
            parse_python_assigns("a=eval('import sys; sys.exit=DR EVIL')")
            self.assertTrue("Function 'eval'" in cm.exception.message)
            self.assertTrue("is not builtin" in cm.exception.message)

        # Python parser fails here anyway
        with self.assertRaises(SyntaxError) as cm:
            parse_python_assigns("1 = 'a'")


    def test_parse_python_assigns_multiple(self):
        """ Test multiple assignments and tuple/list unpacking """
        assign_str = "a,b=[1,2]; c=[1,2]; d=e=f=(1,2,3); g,h=(1,2)"

        D = parse_python_assigns(assign_str)

        self.assertTrue(D == {
            "a" : 1,
            "b" : 2,
            "c" : [1, 2],
            "d" : (1, 2, 3),
            "e" : (1, 2, 3),
            "f" : (1, 2, 3),
            "g" : 1,
            "h" : 2})

        with self.assertRaises(ValueError) as cm:
            parse_python_assigns("a, b = [1,2,3]")
            ex_fragment = ("The number of tuple elements did not "
                            "match the number of values")

            self.assertTrue(ex_fragment in cm.exception.message)

if __name__ == "__main__":
    unittest.main()
