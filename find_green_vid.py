import cv2
import numpy as np
from loguru import logger

# Initialize the logger to log events to a file
logger.add("video_processing.log", format="{time} {level} {message}", level="INFO")

class TrackerHandler:
    """Handles a single tracker."""
    def __init__(self, tracker, frame, bbox):
        self.tracker = tracker
        self.tracker.init(frame, bbox)
        self.bbox = bbox
        self.frames_lost = 0

    def update(self, frame):
        success, box = self.tracker.update(frame)
        if success:
            self.bbox = box
            self.frames_lost = 0
        else:
            self.frames_lost += 1
        return success, box

    def is_lost(self, max_frames_lost=20):
        return self.frames_lost > max_frames_lost

    def is_within_region(self, box, tolerance=10):
        """Check if the box is within a certain tolerance of the original region."""
        x, y, w, h = self.bbox
        nx, ny, nw, nh = box
        return (
            abs(nx - x) < tolerance * w and
            abs(ny - y) < tolerance * h and
            abs(nw - w) < tolerance * w and
            abs(nh - h) < tolerance * h
        )

def initialize_tracker(frame):
    """Prompt the user to select a box to initialize a tracker."""
    bbox = cv2.selectROI("Select Tracker", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Tracker")
    tracker = cv2.TrackerCSRT_create()  # You can also use other trackers like TrackerKCF or TrackerMIL
    return TrackerHandler(tracker, frame, bbox), bbox

def get_hsv_range_for_green():
    """Get HSV range for the color green."""
    # Define HSV range for green color
    lower_bound = np.array([35, 40, 40])  # Lower bound for green (hue, saturation, value)
    upper_bound = np.array([85, 255, 255])  # Upper bound for green
    return lower_bound, upper_bound

def process_video_and_generate_output(video_path, output_path):
    logger.info(f"Starting video processing: {video_path}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        logger.error(f"Error: Could not open video file {video_path}.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}, Total frames: {total_frames}")

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))  # Width is doubled for side-by-side output

    frame_count = 0
    tracker_handler = None  # Will be initialized on the first frame
    lower_hsv, upper_hsv = get_hsv_range_for_green()  # Set the HSV bounds for green
    continue_old_model_frames = 0

    # Read until the video is completed
    while cap.isOpened():
        ret, frame = cap.read()  # Capture frame-by-frame
        frame_count += 1

        if not ret:
            logger.info(f"Reached the end of the video or error occurred at frame {frame_count}.")
            break

        # Log each frame being processed
        logger.info(f"Processing frame {frame_count}/{total_frames}")

        # If this is the first frame, initialize the tracker
        if tracker_handler is None:
            logger.info("Initializing tracker for the first frame.")
            tracker_handler, initial_bbox = initialize_tracker(frame)

        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for the green color range
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Define a kernel for morphological operations (the size of the kernel determines how aggressively close blobs will be merged)
        kernel = np.ones((5, 5), np.uint8)

        # Apply morphological closing to join close blobs
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Apply mask to the original frame to create the filtered frame
        filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Update the tracker
        if continue_old_model_frames > 0:
            success = True
            box = tracker_handler.bbox
            continue_old_model_frames -= 1
        else:
            success, box = tracker_handler.update(filtered_frame)

        if not success:
            if tracker_handler.is_lost():
                # If tracking has been lost for too long, prompt the user to select a new bounding box
                logger.info(f"Tracker lost the object for more than 30 frames at frame {frame_count}. Reinitializing tracker.")
                tracker_handler, new_bbox = initialize_tracker(frame)
                success, box = tracker_handler.update(filtered_frame)
        else:
            # If tracking succeeds, check if the box is within the expected region
            if not tracker_handler.is_within_region(box):
                # If the tracker drifts too far, prompt for a new selection
                logger.info(f"Tracker drifted too far at frame {frame_count}. Reinitializing tracker.")
                tracker_handler, new_bbox = initialize_tracker(frame)
                success, box = tracker_handler.update(filtered_frame)

        if success:
            x, y, w, h = [int(v) for v in box]
            # Draw bounding box on the filtered frame
            cv2.rectangle(filtered_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(filtered_frame, "Tracked Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Draw bounding box on the original (unfiltered) frame as well
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Tracked Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Log tracking information
            logger.info(f"Tracker position - x: {x}, y: {y}, w: {w}, h: {h}")

        # Concatenate the original and filtered frames horizontally (side by side)
        combined_frame = np.hstack((frame, filtered_frame))

        # Write the combined frame to the output video
        out.write(combined_frame)

        # Display the combined frame in real time
        cv2.imshow('Video Processing', combined_frame)

        # Event handling for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space pressed, continue with the old detection model for 5 more frames
            logger.info("Space pressed, continuing with the old detection model for 5 more frames.")
            continue_old_model_frames = 5
        elif key == 13:  # Enter pressed, initialize a new tracker (ASCII code for Enter is 13)
            logger.info("Enter pressed, adding a new ROI.")
            tracker_handler, new_bbox = initialize_tracker(frame)

    # When everything is done, release the video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    logger.info(f"Video processing complete. Output saved at: {output_path}")

# Call the function with an example video path and output path
input_video_path = "./Videos/cropped_3.mp4"
output_video_path = "./output_video.mp4"
process_video_and_generate_output(input_video_path, output_video_path)

print("Video processing complete. Check the log file for details.")
