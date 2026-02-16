"""
Test data generator for creating synthetic images and videos with green blobs.
This module provides utilities to generate controlled test data for green blob detection.
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional


class TestDataGenerator:
    """Generate synthetic test images and videos with green blobs."""

    # Pure green in BGR format
    GREEN_BGR = (0, 255, 0)
    GRAY_BG = (128, 128, 128)

    @staticmethod
    def create_blank_image(width: int = 640, height: int = 480,
                          background_color: Tuple[int, int, int] = GRAY_BG) -> np.ndarray:
        """Create a blank image with specified dimensions and background color."""
        return np.full((height, width, 3), background_color, dtype=np.uint8)

    @staticmethod
    def draw_circle_blob(image: np.ndarray, center: Tuple[int, int],
                        radius: int, color: Tuple[int, int, int] = GREEN_BGR) -> np.ndarray:
        """Draw a filled circular green blob on an image."""
        cv2.circle(image, center, radius, color, -1)
        return image

    @staticmethod
    def create_image_with_blob(width: int = 640, height: int = 480,
                              blob_center: Tuple[int, int] = (320, 240),
                              blob_radius: int = 50,
                              background_color: Tuple[int, int, int] = GRAY_BG) -> Tuple[np.ndarray, dict]:
        """
        Create an image with a single green blob.

        Returns:
            Tuple of (image, ground_truth_dict)
            ground_truth_dict contains: center, radius, bbox (x, y, w, h)
        """
        image = TestDataGenerator.create_blank_image(width, height, background_color)
        TestDataGenerator.draw_circle_blob(image, blob_center, blob_radius)

        # Calculate expected bounding box
        x = blob_center[0] - blob_radius
        y = blob_center[1] - blob_radius
        w = h = blob_radius * 2

        ground_truth = {
            'center': blob_center,
            'radius': blob_radius,
            'bbox': (x, y, w, h),  # (x, y, width, height)
        }

        return image, ground_truth

    @staticmethod
    def create_video_with_moving_blob(output_path: str,
                                     width: int = 640,
                                     height: int = 480,
                                     fps: int = 30,
                                     duration_seconds: float = 3.0,
                                     blob_radius: int = 30,
                                     start_pos: Tuple[int, int] = (100, 240),
                                     end_pos: Tuple[int, int] = (540, 240),
                                     background_color: Tuple[int, int, int] = GRAY_BG) -> dict:
        """
        Create a video with a green blob moving linearly from start to end position.

        Returns:
            Dictionary with video metadata and per-frame ground truth data:
            - path: video file path
            - width, height, fps, num_frames, duration
            - frames: List of dicts with 'center' and 'bbox' for each frame
        """
        num_frames = int(fps * duration_seconds)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frames_data = []

        for frame_idx in range(num_frames):
            # Linear interpolation
            t = frame_idx / (num_frames - 1) if num_frames > 1 else 0
            x = int(start_pos[0] + (end_pos[0] - start_pos[0]) * t)
            y = int(start_pos[1] + (end_pos[1] - start_pos[1]) * t)

            # Create frame
            frame = TestDataGenerator.create_blank_image(width, height, background_color)
            TestDataGenerator.draw_circle_blob(frame, (x, y), blob_radius)

            out.write(frame)

            # Store ground truth for this frame
            bbox_x = x - blob_radius
            bbox_y = y - blob_radius
            bbox_w = bbox_h = blob_radius * 2

            frames_data.append({
                'frame_idx': frame_idx,
                'center': (x, y),
                'radius': blob_radius,
                'bbox': (bbox_x, bbox_y, bbox_w, bbox_h),
            })

        out.release()

        return {
            'path': output_path,
            'width': width,
            'height': height,
            'fps': fps,
            'num_frames': num_frames,
            'duration': duration_seconds,
            'frames': frames_data,
        }

    @staticmethod
    def create_video_circular_motion(output_path: str,
                                    width: int = 640,
                                    height: int = 480,
                                    fps: int = 30,
                                    duration_seconds: float = 2.0,
                                    blob_radius: int = 30,
                                    circle_center: Tuple[int, int] = (320, 240),
                                    circle_radius: int = 100) -> dict:
        """Create a video with green blob moving in a circle."""
        num_frames = int(fps * duration_seconds)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frames_data = []

        for frame_idx in range(num_frames):
            t = frame_idx / num_frames
            angle = 2 * np.pi * t

            x = int(circle_center[0] + circle_radius * np.cos(angle))
            y = int(circle_center[1] + circle_radius * np.sin(angle))

            frame = TestDataGenerator.create_blank_image(width, height)
            TestDataGenerator.draw_circle_blob(frame, (x, y), blob_radius)

            out.write(frame)

            bbox_x = x - blob_radius
            bbox_y = y - blob_radius
            bbox_w = bbox_h = blob_radius * 2

            frames_data.append({
                'frame_idx': frame_idx,
                'center': (x, y),
                'radius': blob_radius,
                'bbox': (bbox_x, bbox_y, bbox_w, bbox_h),
            })

        out.release()

        return {
            'path': output_path,
            'width': width,
            'height': height,
            'fps': fps,
            'num_frames': num_frames,
            'duration': duration_seconds,
            'frames': frames_data,
        }
