import numpy as np
import cv2 as cv
import math
import atexit
import os

class ImageProcessor:
    def __init__(self):
        self.frames = []
        self.cap = None
        atexit.register(self.cleanup)

    def cleanup(self):
        print("Cleaning up...")
        cv.destroyAllWindows()
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                print(f"Failed to release cap: {e}")

        if self.frames:
            try:
                self.save_vid("./recovered_vid.mp4", self.frames, 30)
            except Exception as e:
                print(f"Failed to save recovered frames: {e}")

    def check_color_space(self, frame):
        assert frame is not None, "The frame is None, cannot determine color space."

        if len(frame.shape) == 2:
            return "grayscale"
        elif len(frame.shape) == 3:
            channels = frame.shape[2]
            if channels == 3:
                pixel = frame[0, 0]
                if pixel[0] <= 179 and pixel[1] <= 255 and pixel[2] <= 255:
                    return "HSV or BGR"
                else:
                    return "BGR"
            elif channels == 4:
                return "BGRA"
        return "Unknown color space"

    def open_vid(self, file_path):
        self.cap = cv.VideoCapture(file_path)
        assert self.cap.isOpened(), "Could not open video"

        num_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        while True:
            ret, frame = self.cap.read()
            curr_frame_num = int(self.cap.get(cv.CAP_PROP_POS_FRAMES))

            if curr_frame_num >= num_frames or not ret:
                break

            yield frame

        self.cap.release()

    def save_vid(self, file_path, frames, fps):
        assert len(frames) > 0, "frames to save is empty"

        frame_height, frame_width = frames[0].shape[:2]
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(file_path, fourcc, fps, (frame_width, frame_height), True)

        for frame in frames:
            assert isinstance(frame, np.ndarray), "frame to write is wrong format"
            assert frame.shape[:2] == (frame_height, frame_width), "All frames must have the same size"
            out.write(frame)

        print("saved frames to", file_path)
        out.release()

    def display_frames(self, frames):
        assert frames is not None, "bad input argument to display_frames"
        assert len(frames) > 0, "empty list to display_frames"

        frame_height, frame_width = frames[0].shape[:2]
        converted_frames = []
        for frame in frames:
            assert isinstance(frame, np.ndarray), "All frames should be Numpy arrays."
            assert frame.shape[:2] == (frame_height, frame_width), "All frames don't have the same size"

            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
            converted_frames.append(frame)

        num_frames = len(converted_frames)
        grid_cols = int(math.ceil(math.sqrt(num_frames)))
        grid_rows = int(math.ceil(num_frames / grid_cols))

        blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        while len(converted_frames) < grid_rows * grid_cols:
            converted_frames.append(blank_frame)

        grid_frames = []
        for row in range(grid_rows):
            start_idx = row * grid_cols
            end_idx = start_idx + grid_cols
            row_frames = converted_frames[start_idx:end_idx]
            if row % 2 != 0:
                row_frames = row_frames[::-1]
            grid_frames.append(np.hstack(row_frames))

        grid_canvas = np.vstack(grid_frames)
        cv.imshow('Output Video', grid_canvas)
        cv.waitKey(1)

        return grid_canvas

    def find_contour(self, color_frame, grayscale_frame):
        assert grayscale_frame is not None, "empty grayscale frame given to find_contour"
        assert color_frame is not None, "empty color frame given to find_contour"
        assert self.check_color_space(grayscale_frame) == "grayscale", "frame given to find_contour wasn't grayscale"

        thresh = cv.adaptiveThreshold(grayscale_frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        mod_frame = color_frame.copy()
        for cnt in contours:
            cv.drawContours(image=mod_frame, contours=[cnt], contourIdx=0, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)

        return mod_frame, contours

    def apply_grayscale(self, frame):
        assert frame is not None, "empty frame given to apply_grayscale"
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    def apply_blur(self, frame):
        assert frame is not None, "empty frame given to apply_blur"
        return cv.GaussianBlur(frame, (15, 15), 0)

    def filter_color_range(self, frame, color_bounds):
        assert frame is not None, "Got empty frame when trying to filter color"
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        masks = [cv.inRange(hsv_frame, lower, upper) for lower, upper in color_bounds]
        full_mask = cv.add(masks[0], masks[0])
        for mask in masks[1:]:
            full_mask = cv.add(full_mask, mask)

        return cv.bitwise_and(frame, frame, mask=full_mask)

    def get_fps(self, video_path):
        video = cv.VideoCapture(video_path)
        assert video.isOpened(), "video could not be opened to get fps"
        fps = video.get(cv.CAP_PROP_FPS)
        video.release()

        return fps if fps > 0 else 30

    #uses opencv to create a black video
    def generate_black_video(output_path, width, height, num_frames, fps=30):
        fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec
        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        black_frame = np.zeros((height, width, 3), dtype=np.uint8)

        for _ in range(num_frames):
            out.write(black_frame)

        out.release()


