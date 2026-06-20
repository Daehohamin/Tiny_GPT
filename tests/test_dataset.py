import unittest

from src.dataset import CharSequenceDataset


class DatasetTest(unittest.TestCase):
    def test_x_y_shifting(self):
        ds = CharSequenceDataset([0, 1, 2, 3, 4], block_size=3)
        x, y = ds[0]
        self.assertEqual(x.tolist(), [0, 1, 2])
        self.assertEqual(y.tolist(), [1, 2, 3])
        self.assertEqual(len(ds), 2)


if __name__ == "__main__":
    unittest.main()
