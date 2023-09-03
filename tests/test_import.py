import unittest


class TestPaddleXDEImport(unittest.TestCase):
    def test_import(self):
        import paddlexde

        assert hasattr(paddlexde, "__version__")
