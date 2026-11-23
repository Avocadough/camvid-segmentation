import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from segmentation import build_model, image_pairs, make_dataset, read_palette


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "val").mkdir()
        (self.root / "val_labels").mkdir()
        mask = np.zeros((2, 2, 3), dtype=np.uint8)
        mask[:, 1, 0] = 255
        png = tf.io.encode_png(mask).numpy()
        (self.root / "val/frame.png").write_bytes(png)
        (self.root / "val_labels/frame_L.png").write_bytes(png)
        (self.root / "class_dict.csv").write_text("name,r,g,b\nroad,0,0,0\ncar,255,0,0\n")

    def test_missing_partner(self):
        (self.root / "val_labels/frame_L.png").rename(self.root / "val_labels/other_L.png")
        with self.assertRaisesRegex(ValueError, "missing masks=.*frame"):
            image_pairs(self.root, "val")

    def test_mask_and_model(self):
        pairs = image_pairs(self.root, "val")
        palette, classes = read_palette(self.root / "class_dict.csv")
        image, labels = next(iter(make_dataset(pairs, palette, 8, 1)))
        expected = np.zeros((8, 8), dtype=int)
        expected[:, 4:] = 1
        np.testing.assert_array_equal(labels[0], expected)
        prediction = build_model(8, classes, 2, 1, "leaky_relu")(image)
        self.assertEqual(tuple(prediction.shape), (1, 8, 8, 2))
        np.testing.assert_allclose(tf.reduce_sum(prediction, axis=-1), 1, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
