"""
Green blob detector with standardized output interface.
This module provides a consistent interface for detecting green blobs in images/videos.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List


class GreenBlobDetector:
    """
    Detects green blobs in images and returns bounding boxes.

    This is a wrapper interface that standardizes the output format,
    making it easy to swap detection algorithms.
    """

    def __init__(self,
                 lower_green: np.ndarray = np.array([30, 40, 40]),
                 upper_green: np.ndarray = np.array([90, 255, 255])):
        """
        Initialize the detector with HSV color bounds.

        Args:
            lower_green: Lower HSV bound for green
            upper_green: Upper HSV bound for green
        """
        self.lower_green = lower_green
        self.upper_green = upper_green

    def detect_in_image(self, image: np.ndarray) -> Optional[dict]:
        """
        Detect the largest green blob in an image.

        Args:
            image: BGR image (numpy array)

        Returns:
            Dictionary with detection results or None if no blob found:
            {
                'bbox': (x, y, w, h),  # Bounding box
                'center': (cx, cy),     # Center point
                'area': int,            # Contour area
                'contour': np.ndarray,  # Actual contour points
            }
        """
        if image is None or image.size == 0:
            return None

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create mask for green
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate center
        cx = x + w // 2
        cy = y + h // 2

        return {
            'bbox': (x, y, w, h),
            'center': (cx, cy),
            'area': area,
            'contour': largest_contour,
        }

    def detect_in_video(self, video_path: str) -> List[Optional[dict]]:
        """
        Detect green blob in each frame of a video.

        Args:
            video_path: Path to video file

        Returns:
            List of detection results (one per frame), None for frames with no detection
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        results = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detection = self.detect_in_image(frame)
            results.append(detection)

        cap.release()
        return results

    def visualize_detection(self, image: np.ndarray, detection: dict,
                           color: Tuple[int, int, int] = (0, 255, 0),
                           thickness: int = 2) -> np.ndarray:
        """
        Draw bounding box and center point on image.

        Args:
            image: Input image
            detection: Detection dict from detect_in_image()
            color: BGR color for drawing
            thickness: Line thickness

        Returns:
            Image with visualization overlay
        """
        vis_image = image.copy()

        if detection is None:
            return vis_image

        x, y, w, h = detection['bbox']
        cx, cy = detection['center']

        # Draw bounding box
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)

        # Draw center point
        cv2.circle(vis_image, (cx, cy), 5, color, -1)

        # Draw area text
        cv2.putText(vis_image, f"Area: {detection['area']}",
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, color, 1)

        return vis_image
