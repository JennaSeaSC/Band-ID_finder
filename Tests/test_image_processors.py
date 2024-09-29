import unittest
import os
import sys
import numpy as np

#import image_processors.py file in main dir
from importmonkey import add_path
add_path("../")
from image_processors import ImageProcessor

# Test class for ImageProcessor
class TestImageProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = ImageProcessor()
        cls.test_frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cls.black_video_path = 'black_video.mp4'
        ImageProcessor.generate_black_video(cls.black_video_path, width=100, height=100, num_frames=30, fps=30)

    @classmethod
    def tearDownClass(cls):
        # Clean up the generated video file
        if os.path.exists(cls.black_video_path):
            os.remove(cls.black_video_path)

    def test_check_color_space(self):
        color_space = self.processor.check_color_space(self.test_frame)
        self.assertIn(color_space, ["BGR", "HSV or BGR"])

    def test_apply_grayscale(self):
        gray_frame = self.processor.apply_grayscale(self.test_frame)
        self.assertEqual(len(gray_frame.shape), 2)

    def test_apply_blur(self):
        blurred_frame = self.processor.apply_blur(self.test_frame)
        self.assertEqual(blurred_frame.shape, self.test_frame.shape)

    def test_filter_color_range(self):
        bounds = [(np.array([0, 0, 0]), np.array([255, 255, 255]))]
        filtered_frame = self.processor.filter_color_range(self.test_frame, bounds)
        self.assertEqual(filtered_frame.shape, self.test_frame.shape)

    def test_get_fps(self):
        fps = self.processor.get_fps(self.black_video_path)
        self.assertEqual(fps, 30)

if __name__ == '__main__':
    unittest.main()
