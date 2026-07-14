import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from a7_marker_based import measure_object_from_pixel_coords


class MarkerBasedMeasurementTests(unittest.TestCase):
    def test_measure_object_from_pixel_coords_for_rectangle(self):
        coords = [(0, 0), (20, 0), (20, 10), (0, 10)]
        result = measure_object_from_pixel_coords(coords, pixels_per_metric=2.0)

        self.assertAlmostEqual(result["length"], 10.0)
        self.assertAlmostEqual(result["breadth"], 5.0)
        self.assertAlmostEqual(result["area"], 50.0)


if __name__ == "__main__":
    unittest.main()
