"OpenCV Band Tracker Green"
import cv2
import numpy as np
import os

# Load video
cap = cv2.VideoCapture("C:/Users/jbcon/Documents/CODE/Band-Tracker_Video/TTNT_PTP_FOLLOW__CUT_VIDEO.mp4")

# Define the lower and upper bounds of the color you want to track in HSV
# Example: Green color

lower_green = np.array([30, 40, 40])
upper_green = np.array([90, 255, 255])

# Estimate the approximate area of a 1 cm² square in pixels
# This depends on the camera setup, resolution, and distance
# For example, if 1 cm² ≈ 500 pixels² (adjust based on your setup)
expected_area = 5  # Adjust this value according to your video and objects

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the color
    color_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # Find contours in the color mask
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the detected colored objects
    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter out contours based on the expected area of 1 cm²
        if expected_area * 0.8 < area < expected_area * 1.2:  # 20% tolerance
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Optionally, you can add text to display the area
            # cv2.putText(frame, f"Area: {int(area)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show the video with tracking
    cv2.imshow('Frame', frame)
    cv2.imshow('Color Mask', color_mask)

    # Exit when 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()