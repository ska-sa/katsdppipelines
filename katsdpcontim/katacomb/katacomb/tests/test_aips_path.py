from contextlib import contextmanager
import unittest

from katacomb import obit_context, uv_factory, AIPSPath
from katacomb.aips_path import next_seq_nr

@contextmanager
def file_cleaner(paths):
    try:
        for path in paths:
            with uv_factory(aips_path=path, mode="w") as f:
                f.Zap()

        yield
    finally:
        for path in paths:
            with uv_factory(aips_path=path, mode="w") as f:
                f.Zap()

class TestAipsPath(unittest.TestCase):

    def test_next_seq_nr(self):
        """ Test finding the next highest disk sequence number of an AIPS Path """

        # Create two AIPS paths, one with with sequence number 10 and 11
        p1 = AIPSPath(name='test', disk=1, aclass="klass", seq=10)
        p2  = p1.copy(seq=11)

        with obit_context(), file_cleaner([p1, p2]) as fc:
            # Create the first file and test highest sequence number
            with uv_factory(aips_path=p1, mode="w") as uvc1: pass
            self.assertTrue(next_seq_nr(p1) == 11)

            # Create the second file and test highest sequence number
            with uv_factory(aips_path=p2, mode="w") as uvc2: pass
            self.assertTrue(next_seq_nr(p1) == 12)
            self.assertTrue(next_seq_nr(p2) == 12)
