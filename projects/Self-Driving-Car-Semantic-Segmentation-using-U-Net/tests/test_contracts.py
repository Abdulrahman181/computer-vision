"""Contract tests that run without TensorFlow, OpenCV, or Streamlit."""
import unittest

import numpy as np

from segmentation_utils import (
    class_coverage,
    prediction_to_probabilities,
    validate_class_mask,
    validate_model_input_shape,
    validate_model_output_shape,
)


class SegmentationContractTests(unittest.TestCase):
    def test_logits_are_normalized_to_probabilities(self) -> None:
        probabilities = prediction_to_probabilities(np.array([[[[2.0, 0.0], [0.0, 2.0]]]]), 2)
        np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0)
        self.assertGreater(float(probabilities[0, 0, 0, 0]), 0.8)

    def test_probability_tensor_is_preserved(self) -> None:
        original = np.array([[[[0.25, 0.75]]]], dtype=np.float32)
        np.testing.assert_allclose(prediction_to_probabilities(original, 2), original)

    def test_model_shapes_reject_wrong_contracts(self) -> None:
        self.assertEqual(validate_model_input_shape((None, 256, 256, 3))[-1], 3)
        self.assertEqual(validate_model_output_shape((None, None, None, 8), 8)[-1], 8)
        with self.assertRaises(ValueError):
            validate_model_input_shape((None, 256, 256, 1))
        with self.assertRaises(ValueError):
            validate_model_output_shape((None, 256, 256, 7), 8)

    def test_class_coverage_rejects_out_of_range_values(self) -> None:
        mask = np.array([[0, 1], [1, 1]], dtype=np.int32)
        self.assertEqual(class_coverage(mask, ("background", "road")), {"background": 25.0, "road": 75.0})
        with self.assertRaises(ValueError):
            validate_class_mask(np.array([[0, 2]], dtype=np.int32), 2)


if __name__ == "__main__":
    unittest.main()
