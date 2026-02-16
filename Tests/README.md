# Band-ID Finder Test Suite

Comprehensive test harness for green blob detection with synthetic test data.

## Test Files

### `test_green_detection.py`
Core unit tests for blob detection and tracking:
- **TestGreenBlobDetection**: Tests detection in single images
  - Various positions (corners, center)
  - Different blob sizes (small, medium, large)
  - Edge cases (no blob, invalid input)

- **TestGreenBlobTracking**: Tests tracking in videos
  - Horizontal, vertical, and circular motion
  - Fast motion scenarios
  - Trajectory accuracy validation

- **TestGreenBlobVisualization**: Tests visualization utilities

### `test_artifacts.py`
Generates visual artifacts for reviewing test results:
- Detection sample grid showing all positions
- Trajectory plots (position over time, error graphs)
- Annotated videos with bounding boxes
- Performance summary charts

### `test_data_generator.py`
Utilities for creating synthetic test data:
- Generate images with green blobs at specified positions
- Create videos with moving blobs (linear, circular motion)
- Provides ground truth data (center, bbox) for validation

## Running Tests

### Run all unit tests:
```bash
# Using pytest
pytest Tests/test_green_detection.py -v

# Using unittest
python -m unittest Tests.test_green_detection -v
```

### Generate visual artifacts:
```bash
# Run artifact generator
python Tests/test_artifacts.py -v

# Artifacts will be saved to Tests/artifacts/
# - detection_samples.png
# - trajectory_plot.png
# - circular_trajectory.png
# - accuracy_summary.png
# - input_video.mp4 (original test video)
# - output_video_annotated.mp4 (with detection overlays)
```

### Run from Docker:
```bash
# Build image
docker build -t band-id-finder .

# Run tests
docker run band-id-finder pytest Tests/test_green_detection.py -v

# Generate artifacts (mount volume to get outputs)
docker run -v $(pwd)/Tests/artifacts:/app/Tests/artifacts band-id-finder python Tests/test_artifacts.py -v
```

## Test Data Format

### Detection Output
The `GreenBlobDetector` returns a standardized dict:
```python
{
    'bbox': (x, y, w, h),    # Bounding box
    'center': (cx, cy),       # Center point
    'area': int,              # Contour area
    'contour': np.ndarray,    # Contour points
}
```

### Ground Truth Format
Test data generator provides ground truth:
```python
{
    'center': (x, y),
    'radius': int,
    'bbox': (x, y, w, h),
}
```

## Swapping Detection Algorithms

The test suite is algorithm-agnostic. To test a different detector:

1. Implement the interface in `green_detector.py`:
   ```python
   def detect_in_image(self, image: np.ndarray) -> Optional[dict]:
       # Your algorithm here
       return {
           'bbox': (x, y, w, h),
           'center': (cx, cy),
           'area': area,
           'contour': contour,
       }
   ```

2. Run the test suite - it validates the interface contract, not implementation

3. Generate artifacts to visually compare results

## Test Metrics

Tests validate:
- **Detection Rate**: % of frames where blob is found
- **Position Accuracy**: Distance between detected and expected center (should be < 5px)
- **Bounding Box Accuracy**: BBox coordinates within tolerance
- **Tracking Continuity**: No dropped frames during motion

## Adding New Tests

```python
def test_your_scenario(self):
    # Generate test data with known ground truth
    image, ground_truth = TestDataGenerator.create_image_with_blob(
        blob_center=(320, 240),
        blob_radius=50
    )

    # Run your detector
    detection = self.detector.detect_in_image(image)

    # Validate against ground truth
    self.assertIsNotNone(detection)
    self.assertCenterClose(detection['center'], ground_truth['center'])
```
