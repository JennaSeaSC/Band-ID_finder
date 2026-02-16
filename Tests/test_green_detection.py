"""
Comprehensive unit tests for green blob detection.

These tests use synthetic data with known ground truth to validate detection algorithms.
The tests are algorithm-agnostic - swap out the detector implementation as needed.
"""
import unittest
import os
import sys
import numpy as np
import cv2
import tempfile
import shutil

# Import the test data generator and detector
from test_data_generator import TestDataGenerator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from green_detector import GreenBlobDetector


class TestGreenBlobDetection(unittest.TestCase):
    """Test suite for green blob detection in images."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.detector = GreenBlobDetector()
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def assertBBoxClose(self, detected_bbox, expected_bbox, tolerance=5):
        """Assert that detected bounding box is close to expected."""
        d_x, d_y, d_w, d_h = detected_bbox
        e_x, e_y, e_w, e_h = expected_bbox

        self.assertAlmostEqual(d_x, e_x, delta=tolerance,
                              msg=f"BBox x mismatch: {d_x} vs {e_x}")
        self.assertAlmostEqual(d_y, e_y, delta=tolerance,
                              msg=f"BBox y mismatch: {d_y} vs {e_y}")
        self.assertAlmostEqual(d_w, e_w, delta=tolerance,
                              msg=f"BBox width mismatch: {d_w} vs {e_w}")
        self.assertAlmostEqual(d_h, e_h, delta=tolerance,
                              msg=f"BBox height mismatch: {d_h} vs {e_h}")

    def assertCenterClose(self, detected_center, expected_center, tolerance=5):
        """Assert that detected center is close to expected."""
        d_x, d_y = detected_center
        e_x, e_y = expected_center

        self.assertAlmostEqual(d_x, e_x, delta=tolerance,
                              msg=f"Center x mismatch: {d_x} vs {e_x}")
        self.assertAlmostEqual(d_y, e_y, delta=tolerance,
                              msg=f"Center y mismatch: {d_y} vs {e_y}")

    # ===== BASIC IMAGE TESTS =====

    def test_detect_centered_blob(self):
        """Test detection of centered green blob."""
        image, ground_truth = TestDataGenerator.create_image_with_blob(
            width=640, height=480,
            blob_center=(320, 240),
            blob_radius=50
        )

        detection = self.detector.detect_in_image(image)

        self.assertIsNotNone(detection, "Failed to detect blob")
        self.assertBBoxClose(detection['bbox'], ground_truth['bbox'])
        self.assertCenterClose(detection['center'], ground_truth['center'])
        self.assertGreater(detection['area'], 0)

    def test_detect_small_blob(self):
        """Test detection of small green blob (15px radius)."""
        image, ground_truth = TestDataGenerator.create_image_with_blob(
            blob_center=(200, 150),
            blob_radius=15
        )

        detection = self.detector.detect_in_image(image)

        self.assertIsNotNone(detection, "Failed to detect small blob")
        self.assertCenterClose(detection['center'], ground_truth['center'])

    def test_detect_large_blob(self):
        """Test detection of large green blob (100px radius)."""
        image, ground_truth = TestDataGenerator.create_image_with_blob(
            blob_center=(320, 240),
            blob_radius=100
        )

        detection = self.detector.detect_in_image(image)

        self.assertIsNotNone(detection, "Failed to detect large blob")
        self.assertCenterClose(detection['center'], ground_truth['center'])
        self.assertGreater(detection['area'], 30000, "Area too small for large blob")

    def test_detect_blob_top_left(self):
        """Test detection of blob in top-left corner."""
        image, ground_truth = TestDataGenerator.create_image_with_blob(
            blob_center=(50, 50),
            blob_radius=30
        )

        detection = self.detector.detect_in_image(image)

        self.assertIsNotNone(detection, "Failed to detect top-left blob")
        self.assertCenterClose(detection['center'], ground_truth['center'])

    def test_detect_blob_bottom_right(self):
        """Test detection of blob in bottom-right corner."""
        image, ground_truth = TestDataGenerator.create_image_with_blob(
            width=640, height=480,
            blob_center=(590, 430),
            blob_radius=30
        )

        detection = self.detector.detect_in_image(image)

        self.assertIsNotNone(detection, "Failed to detect bottom-right blob")
        self.assertCenterClose(detection['center'], ground_truth['center'])

    def test_no_blob_in_empty_image(self):
        """Test that empty image returns None."""
        image = TestDataGenerator.create_blank_image()

        detection = self.detector.detect_in_image(image)

        self.assertIsNone(detection, "Should not detect blob in empty image")

    def test_invalid_image_returns_none(self):
        """Test that None/empty image returns None."""
        self.assertIsNone(self.detector.detect_in_image(None))
        self.assertIsNone(self.detector.detect_in_image(np.array([])))

    def test_detection_has_required_fields(self):
        """Test that detection dict contains all required fields."""
        image, _ = TestDataGenerator.create_image_with_blob()

        detection = self.detector.detect_in_image(image)

        self.assertIn('bbox', detection)
        self.assertIn('center', detection)
        self.assertIn('area', detection)
        self.assertIn('contour', detection)

        # Validate types
        self.assertIsInstance(detection['bbox'], tuple)
        self.assertEqual(len(detection['bbox']), 4)
        self.assertIsInstance(detection['center'], tuple)
        self.assertEqual(len(detection['center']), 2)
        self.assertIsInstance(detection['area'], (int, float))
        self.assertIsInstance(detection['contour'], np.ndarray)


class TestGreenBlobTracking(unittest.TestCase):
    """Test suite for tracking green blobs in videos."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.detector = GreenBlobDetector()
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_track_horizontal_motion(self):
        """Test tracking blob moving horizontally across frame."""
        video_path = os.path.join(self.temp_dir, 'horizontal_motion.mp4')
        video_data = TestDataGenerator.create_video_with_moving_blob(
            output_path=video_path,
            fps=30,
            duration_seconds=2.0,
            blob_radius=30,
            start_pos=(100, 240),
            end_pos=(540, 240)
        )

        detections = self.detector.detect_in_video(video_path)

        # Should detect blob in all frames
        self.assertEqual(len(detections), video_data['num_frames'])

        # Check a few frames
        for frame_idx in [0, len(detections) // 2, -1]:
            detection = detections[frame_idx]
            ground_truth = video_data['frames'][frame_idx]

            self.assertIsNotNone(detection,
                                f"Failed to detect blob in frame {frame_idx}")

            # Center should be close to ground truth
            detected_center = detection['center']
            expected_center = ground_truth['center']
            dx = abs(detected_center[0] - expected_center[0])
            dy = abs(detected_center[1] - expected_center[1])

            self.assertLess(dx, 5, f"Frame {frame_idx}: X center off by {dx}px")
            self.assertLess(dy, 5, f"Frame {frame_idx}: Y center off by {dy}px")

    def test_track_vertical_motion(self):
        """Test tracking blob moving vertically."""
        video_path = os.path.join(self.temp_dir, 'vertical_motion.mp4')
        video_data = TestDataGenerator.create_video_with_moving_blob(
            output_path=video_path,
            fps=30,
            duration_seconds=2.0,
            blob_radius=30,
            start_pos=(320, 100),
            end_pos=(320, 380)
        )

        detections = self.detector.detect_in_video(video_path)

        self.assertEqual(len(detections), video_data['num_frames'])

        # Verify detection in all frames
        for detection in detections:
            self.assertIsNotNone(detection, "Lost tracking in a frame")

    def test_track_circular_motion(self):
        """Test tracking blob moving in a circle."""
        video_path = os.path.join(self.temp_dir, 'circular_motion.mp4')
        video_data = TestDataGenerator.create_video_circular_motion(
            output_path=video_path,
            fps=30,
            duration_seconds=2.0,
            blob_radius=25,
            circle_center=(320, 240),
            circle_radius=100
        )

        detections = self.detector.detect_in_video(video_path)

        self.assertEqual(len(detections), video_data['num_frames'])

        # Check tracking accuracy
        num_successful_detections = sum(1 for d in detections if d is not None)
        detection_rate = num_successful_detections / len(detections)

        self.assertGreater(detection_rate, 0.95,
                          f"Detection rate too low: {detection_rate:.2%}")

    def test_track_fast_motion(self):
        """Test tracking fast-moving blob."""
        video_path = os.path.join(self.temp_dir, 'fast_motion.mp4')
        video_data = TestDataGenerator.create_video_with_moving_blob(
            output_path=video_path,
            fps=30,
            duration_seconds=0.5,  # Short duration = fast motion
            blob_radius=30,
            start_pos=(50, 240),
            end_pos=(590, 240)
        )

        detections = self.detector.detect_in_video(video_path)

        # Should maintain tracking even with fast motion
        num_successful = sum(1 for d in detections if d is not None)
        self.assertGreater(num_successful, len(detections) * 0.9,
                          "Lost tracking during fast motion")

    def test_video_trajectory_accuracy(self):
        """Test that tracked trajectory matches ground truth."""
        video_path = os.path.join(self.temp_dir, 'trajectory_test.mp4')
        video_data = TestDataGenerator.create_video_with_moving_blob(
            output_path=video_path,
            fps=30,
            duration_seconds=1.0,
            blob_radius=30,
            start_pos=(100, 240),
            end_pos=(540, 240)
        )

        detections = self.detector.detect_in_video(video_path)

        # Calculate average position error
        total_error = 0
        for detection, frame_data in zip(detections, video_data['frames']):
            if detection:
                detected_center = detection['center']
                expected_center = frame_data['center']
                error = np.sqrt((detected_center[0] - expected_center[0])**2 +
                               (detected_center[1] - expected_center[1])**2)
                total_error += error

        avg_error = total_error / len(detections)
        self.assertLess(avg_error, 5.0,
                       f"Average tracking error too high: {avg_error:.2f}px")


class TestGreenBlobVisualization(unittest.TestCase):
    """Test visualization utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = GreenBlobDetector()

    def test_visualize_detection(self):
        """Test that visualization adds bounding box to image."""
        image, ground_truth = TestDataGenerator.create_image_with_blob(
            blob_center=(320, 240),
            blob_radius=50
        )

        detection = self.detector.detect_in_image(image)
        vis_image = self.detector.visualize_detection(image, detection)

        # Should return an image of same size
        self.assertEqual(vis_image.shape, image.shape)

        # Should be different from original (has overlay)
        self.assertFalse(np.array_equal(vis_image, image))

    def test_visualize_no_detection(self):
        """Test visualization with no detection."""
        image = TestDataGenerator.create_blank_image()

        detection = self.detector.detect_in_image(image)
        vis_image = self.detector.visualize_detection(image, detection)

        # Should return image unchanged
        self.assertTrue(np.array_equal(vis_image, image))


if __name__ == '__main__':
    unittest.main(verbosity=2)
