# Band-ID Finder - Testing & CI/CD Guide

## 🎯 What's Been Set Up

### 1. **Docker Configuration** (`Dockerfile`)
- Python 3.11 slim base
- OpenCV system dependencies pre-installed
- Runs tests by default
- Ready for CI/CD pipelines

### 2. **GitHub Workflows** (`.github/workflows/`)
- **`docker-image.yml`**: Builds Docker image, runs pytest and unittest
- **`codeql.yml`**: Security scanning (weekly + on push/PR)
- **`python-app.yml`**: Python app testing with OpenCV dependencies

### 3. **Test Harness** (`Tests/`)

#### Core Components:
- **`test_data_generator.py`**: Creates synthetic test data
  - Images with green blobs at known positions
  - Videos with moving blobs (linear & circular motion)
  - Provides ground truth (center, bbox, radius)

- **`green_detector.py`**: Standardized detection interface
  - `detect_in_image()`: Returns bbox, center, area, contour
  - `detect_in_video()`: Batch process video frames
  - `visualize_detection()`: Draw bounding boxes
  - **Swap algorithms easily** - just implement the same interface!

- **`test_green_detection.py`**: Comprehensive unit tests
  - 15+ test cases covering various scenarios
  - Validates bounding box accuracy (tolerance: 5px)
  - Tests tracking continuity and accuracy

- **`test_artifacts.py`**: Visual validation artifacts
  - Detection sample grids
  - Trajectory plots with error analysis
  - Annotated videos (input + output with overlays)
  - Performance summary charts
  - **All artifacts tagged with git commit ID and timestamp**

## 🚀 Quick Start

### Run Tests Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run unit tests
pytest Tests/test_green_detection.py -v

# Generate visual artifacts (saved to Tests/artifacts/<timestamp>_<commit>/
python Tests/test_artifacts.py -v
```

### Run Tests in Docker
```bash
# Build image
docker build -t band-id-finder .

# Run tests
docker run band-id-finder

# Generate artifacts with volume mount
docker run -v $(pwd)/Tests/artifacts:/app/Tests/artifacts band-id-finder python Tests/test_artifacts.py
```

### View Results
After running `test_artifacts.py`, check:
```
Tests/artifacts/<timestamp>_<commit>/
├── detection_samples.png          # Grid showing detection at 9 positions
├── trajectory_plot.png             # Position over time + error graph
├── circular_trajectory.png         # 2D trajectory visualization
├── accuracy_summary.png            # Performance metrics bar charts
├── input_video.mp4                 # Original test video
└── output_video_annotated.mp4     # With detection overlays
```

## 🔄 Swapping Detection Algorithms

The test suite is **algorithm-agnostic**. To test a new detector:

1. **Edit `green_detector.py`** - modify the detection logic in `detect_in_image()`:
   ```python
   def detect_in_image(self, image: np.ndarray) -> Optional[dict]:
       # YOUR NEW ALGORITHM HERE
       # ... detection logic ...

       return {
           'bbox': (x, y, w, h),
           'center': (cx, cy),
           'area': area,
           'contour': contour,
       }
   ```

2. **Run tests** to validate:
   ```bash
   pytest Tests/test_green_detection.py -v
   ```

3. **Generate artifacts** to visually compare:
   ```bash
   python Tests/test_artifacts.py -v
   ```

4. **Compare results** using commit IDs in artifact directories

## 📊 Test Metrics

Tests validate:
- ✅ **Detection Rate**: >95% of frames should detect blob
- ✅ **Position Accuracy**: <5px error from ground truth
- ✅ **Bounding Box Accuracy**: Within tolerance for all edges
- ✅ **Tracking Continuity**: No dropped frames during motion

## 🏗️ CI/CD Pipeline

### On Push/PR to `main`:
1. Docker image is built
2. Tests run in container (both pytest & unittest)
3. CodeQL security scan runs
4. Python app tests run with OpenCV

### Failed Tests:
- Check GitHub Actions logs
- Look for assertion errors showing expected vs actual values
- Generate artifacts locally to debug visually

## 📝 Adding New Tests

```python
def test_your_scenario(self):
    """Test description."""
    # Generate synthetic test data
    image, ground_truth = TestDataGenerator.create_image_with_blob(
        blob_center=(320, 240),
        blob_radius=50
    )

    # Run detector
    detection = self.detector.detect_in_image(image)

    # Validate
    self.assertIsNotNone(detection)
    self.assertCenterClose(detection['center'], ground_truth['center'])
    self.assertBBoxClose(detection['bbox'], ground_truth['bbox'])
```

## 🎨 Artifact Examples

All artifacts include:
- **Commit ID** in title/overlay
- **Timestamp** for tracking runs over time
- **Ground truth overlay** (blue) vs **Detection** (green)
- **Error metrics** showing tracking accuracy

This lets you:
- Compare algorithm versions side-by-side
- Track improvements over git history
- Identify regression issues visually
- Share results with team/stakeholders

## 📦 Dependencies

- `opencv-python>=4.5`: Computer vision
- `numpy>=1.23`: Array operations
- `pytest>=8.3.4`: Testing framework
- `matplotlib>=3.5.0`: Plotting artifacts
- `importmonkey>=0.3`: Path manipulation for tests

## 🐛 Troubleshooting

**Tests fail with "module not found"**:
- Run from project root: `pytest Tests/`
- Check PYTHONPATH is set correctly

**OpenCV errors in Docker**:
- System dependencies included in Dockerfile
- Rebuild image: `docker build --no-cache -t band-id-finder .`

**Artifacts not generated**:
- Check matplotlib is installed: `pip install matplotlib`
- Check write permissions on Tests/artifacts/

**Git commit ID shows "no-git"**:
- Ensure you're in a git repository
- Run `git rev-parse HEAD` to verify git works
