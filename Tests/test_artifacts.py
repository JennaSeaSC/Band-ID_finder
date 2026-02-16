"""
Generate test artifacts: plots, visualizations, and annotated videos.
Run this to create visual outputs for reviewing test results.
"""
import unittest
import os
import sys
import numpy as np
import cv2
import tempfile
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Import test utilities
from test_data_generator import TestDataGenerator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.green_detector import GreenBlobDetector

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plotting will be skipped.")


class TestArtifactGenerator(unittest.TestCase):
    """Generate visual artifacts for test validation."""

    @classmethod
    def setUpClass(cls):
        """Set up output directories."""
        cls.detector = GreenBlobDetector()

        # Get git commit ID
        try:
            commit_id = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                cwd=os.path.dirname(os.path.dirname(__file__)),
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
        except:
            commit_id = 'no-git'

        cls.commit_id = commit_id
        cls.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create artifacts directory with commit ID
        cls.artifacts_dir = os.path.join(
            os.path.dirname(__file__),
            'artifacts',
            f'{cls.timestamp}_{cls.commit_id}'
        )
        os.makedirs(cls.artifacts_dir, exist_ok=True)
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
        print(f"\nArtifacts saved to: {cls.artifacts_dir}")

    def test_generate_detection_samples(self):
        """Generate sample images showing detection at different positions."""
        if not HAS_MATPLOTLIB:
            self.skipTest("matplotlib not available")

        positions = [
            (160, 120, "top-left"),
            (320, 120, "top-center"),
            (480, 120, "top-right"),
            (160, 240, "center-left"),
            (320, 240, "center"),
            (480, 240, "center-right"),
            (160, 360, "bottom-left"),
            (320, 360, "bottom-center"),
            (480, 360, "bottom-right"),
        ]

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Green Blob Detection at Various Positions\nCommit: {self.commit_id}', fontsize=16)

        for idx, (x, y, label) in enumerate(positions):
            ax = axes[idx // 3, idx % 3]

            # Create test image
            image, ground_truth = TestDataGenerator.create_image_with_blob(
                blob_center=(x, y),
                blob_radius=30
            )

            # Detect
            detection = self.detector.detect_in_image(image)

            # Visualize
            if detection:
                vis_image = self.detector.visualize_detection(image, detection)
                # Convert BGR to RGB for matplotlib
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            else:
                vis_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            ax.imshow(vis_image)
            ax.set_title(f'{label}\nExpected: {ground_truth["center"]}\n'
                        f'Detected: {detection["center"] if detection else "None"}')
            ax.axis('off')

        plt.tight_layout()
        output_path = os.path.join(self.artifacts_dir, 'detection_samples.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_path}")
        self.assertTrue(os.path.exists(output_path))

    def test_generate_trajectory_plot(self):
        """Generate plot showing detected vs expected trajectory."""
        if not HAS_MATPLOTLIB:
            self.skipTest("matplotlib not available")

        # Create test video
        video_path = os.path.join(self.temp_dir, 'trajectory_test.mp4')
        video_data = TestDataGenerator.create_video_with_moving_blob(
            output_path=video_path,
            fps=30,
            duration_seconds=2.0,
            blob_radius=30,
            start_pos=(100, 240),
            end_pos=(540, 240)
        )

        # Detect in video
        detections = self.detector.detect_in_video(video_path)

        # Extract positions
        expected_x = [frame['center'][0] for frame in video_data['frames']]
        expected_y = [frame['center'][1] for frame in video_data['frames']]
        detected_x = [d['center'][0] if d else None for d in detections]
        detected_y = [d['center'][1] if d else None for d in detections]
        frames = list(range(len(detections)))

        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'Blob Tracking Accuracy - Horizontal Motion\nCommit: {self.commit_id}', fontsize=16)

        # X position over time
        ax1.plot(frames, expected_x, 'b-', label='Expected X', linewidth=2)
        ax1.plot(frames, detected_x, 'r--', label='Detected X', linewidth=1.5)
        ax1.set_ylabel('X Position (px)')
        ax1.set_title('Horizontal Position')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Y position over time
        ax2.plot(frames, expected_y, 'b-', label='Expected Y', linewidth=2)
        ax2.plot(frames, detected_y, 'r--', label='Detected Y', linewidth=1.5)
        ax2.set_ylabel('Y Position (px)')
        ax2.set_title('Vertical Position')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Position error
        errors = []
        for i in range(len(detections)):
            if detections[i]:
                ex, ey = video_data['frames'][i]['center']
                dx, dy = detections[i]['center']
                error = np.sqrt((ex - dx)**2 + (ey - dy)**2)
                errors.append(error)
            else:
                errors.append(None)

        ax3.plot(frames, errors, 'g-', linewidth=2)
        ax3.set_xlabel('Frame Number')
        ax3.set_ylabel('Position Error (px)')
        ax3.set_title('Tracking Error')
        ax3.grid(True, alpha=0.3)

        # Add statistics
        valid_errors = [e for e in errors if e is not None]
        if valid_errors:
            mean_error = np.mean(valid_errors)
            max_error = np.max(valid_errors)
            ax3.axhline(y=mean_error, color='r', linestyle='--',
                       label=f'Mean: {mean_error:.2f}px')
            ax3.legend()

        plt.tight_layout()
        output_path = os.path.join(self.artifacts_dir, 'trajectory_plot.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_path}")
        self.assertTrue(os.path.exists(output_path))

    def test_generate_circular_trajectory_plot(self):
        """Generate 2D plot of circular motion tracking."""
        if not HAS_MATPLOTLIB:
            self.skipTest("matplotlib not available")

        # Create circular motion video
        video_path = os.path.join(self.temp_dir, 'circular_test.mp4')
        video_data = TestDataGenerator.create_video_circular_motion(
            output_path=video_path,
            fps=30,
            duration_seconds=2.0,
            blob_radius=25,
            circle_center=(320, 240),
            circle_radius=100
        )

        # Detect
        detections = self.detector.detect_in_video(video_path)

        # Extract positions
        expected_x = [frame['center'][0] for frame in video_data['frames']]
        expected_y = [frame['center'][1] for frame in video_data['frames']]
        detected_x = [d['center'][0] if d else None for d in detections]
        detected_y = [d['center'][1] if d else None for d in detections]

        # Create 2D trajectory plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(expected_x, expected_y, 'b-', label='Expected Path',
                linewidth=3, alpha=0.5)
        ax.plot(detected_x, detected_y, 'r--', label='Detected Path',
                linewidth=2)
        ax.scatter(expected_x[0], expected_y[0], c='green', s=100,
                  marker='o', label='Start', zorder=5)
        ax.scatter(expected_x[-1], expected_y[-1], c='red', s=100,
                  marker='s', label='End', zorder=5)

        ax.set_xlabel('X Position (px)')
        ax.set_ylabel('Y Position (px)')
        ax.set_title(f'Circular Motion Tracking\nCommit: {self.commit_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Image coordinates

        plt.tight_layout()
        output_path = os.path.join(self.artifacts_dir, 'circular_trajectory.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_path}")
        self.assertTrue(os.path.exists(output_path))

    def test_generate_annotated_video(self):
        """Generate video with detection annotations."""
        # Create test video - save to artifacts as input
        input_artifact_path = os.path.join(self.artifacts_dir, 'input_video.mp4')
        video_data = TestDataGenerator.create_video_with_moving_blob(
            output_path=input_artifact_path,
            fps=30,
            duration_seconds=2.0,
            blob_radius=30,
            start_pos=(100, 240),
            end_pos=(540, 240)
        )

        print(f"Saved input video: {input_artifact_path}")

        # Process and annotate
        output_path = os.path.join(self.artifacts_dir, 'output_video_annotated.mp4')
        cap = cv2.VideoCapture(input_artifact_path)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect
            detection = self.detector.detect_in_image(frame)
            ground_truth = video_data['frames'][frame_idx]

            # Annotate frame
            annotated = frame.copy()

            # Draw ground truth in blue
            gt_x, gt_y = ground_truth['center']
            cv2.circle(annotated, (gt_x, gt_y), 5, (255, 0, 0), -1)
            cv2.putText(annotated, 'Expected', (gt_x + 10, gt_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Draw detection in green
            if detection:
                annotated = self.detector.visualize_detection(annotated, detection)
                error = np.sqrt((detection['center'][0] - gt_x)**2 +
                              (detection['center'][1] - gt_y)**2)
                cv2.putText(annotated, f'Error: {error:.1f}px',
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 0), 2)
            else:
                cv2.putText(annotated, 'NO DETECTION', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Frame number and commit ID
            cv2.putText(annotated, f'Frame: {frame_idx}', (10, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated, f'Commit: {self.commit_id}', (10, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            out.write(annotated)
            frame_idx += 1

        cap.release()
        out.release()

        print(f"Saved output video: {output_path}")
        self.assertTrue(os.path.exists(input_artifact_path))
        self.assertTrue(os.path.exists(output_path))

    def test_generate_accuracy_summary(self):
        """Generate comprehensive accuracy summary plot."""
        if not HAS_MATPLOTLIB:
            self.skipTest("matplotlib not available")

        # Test different scenarios
        test_configs = [
            {'name': 'Small Blob', 'radius': 15, 'speed': 'normal'},
            {'name': 'Medium Blob', 'radius': 30, 'speed': 'normal'},
            {'name': 'Large Blob', 'radius': 50, 'speed': 'normal'},
            {'name': 'Fast Motion', 'radius': 30, 'speed': 'fast'},
        ]

        results = []

        for config in test_configs:
            video_path = os.path.join(self.temp_dir, f'{config["name"]}.mp4')

            duration = 0.5 if config['speed'] == 'fast' else 2.0

            video_data = TestDataGenerator.create_video_with_moving_blob(
                output_path=video_path,
                fps=30,
                duration_seconds=duration,
                blob_radius=config['radius'],
                start_pos=(100, 240),
                end_pos=(540, 240)
            )

            detections = self.detector.detect_in_video(video_path)

            # Calculate metrics
            errors = []
            for det, frame_data in zip(detections, video_data['frames']):
                if det:
                    ex, ey = frame_data['center']
                    dx, dy = det['center']
                    error = np.sqrt((ex - dx)**2 + (ey - dy)**2)
                    errors.append(error)

            detection_rate = sum(1 for d in detections if d) / len(detections)
            mean_error = np.mean(errors) if errors else float('inf')
            max_error = np.max(errors) if errors else float('inf')

            results.append({
                'name': config['name'],
                'detection_rate': detection_rate,
                'mean_error': mean_error,
                'max_error': max_error,
            })

        # Create summary plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Detection Algorithm Performance Summary\nCommit: {self.commit_id} | {self.timestamp}', fontsize=16)

        names = [r['name'] for r in results]
        detection_rates = [r['detection_rate'] * 100 for r in results]
        mean_errors = [r['mean_error'] for r in results]

        # Detection rate
        ax1.bar(names, detection_rates, color='steelblue')
        ax1.set_ylabel('Detection Rate (%)')
        ax1.set_title('Detection Success Rate')
        ax1.set_ylim([0, 105])
        ax1.axhline(y=95, color='r', linestyle='--', label='95% threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Mean error
        ax2.bar(names, mean_errors, color='coral')
        ax2.set_ylabel('Mean Position Error (px)')
        ax2.set_title('Tracking Accuracy')
        ax2.axhline(y=5, color='r', linestyle='--', label='5px threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = os.path.join(self.artifacts_dir, 'accuracy_summary.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_path}")
        print("\nPerformance Summary:")
        for r in results:
            print(f"  {r['name']}:")
            print(f"    Detection Rate: {r['detection_rate']*100:.1f}%")
            print(f"    Mean Error: {r['mean_error']:.2f}px")
            print(f"    Max Error: {r['max_error']:.2f}px")

        self.assertTrue(os.path.exists(output_path))


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
