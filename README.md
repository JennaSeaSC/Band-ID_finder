# Band-ID Finder

A computer vision project for detecting and tracking colored bands in images and videos using OpenCV.

## 🎯 Features

- **Green blob detection** with configurable HSV ranges
- **Video tracking** with trajectory analysis
- **Synthetic test data generation** for algorithm validation
- **Comprehensive test suite** with visual artifacts
- **Docker support** for consistent CI/CD environments
- **GitHub Actions workflows** for automated testing and security scanning

## 📋 Prerequisites

- **Python 3.11+**: Ensure Python is installed and accessible
- **Docker** (optional): For containerized testing
- **Git**: For version control

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/JennaSeaSC/Band-ID_finder.git
cd Band-ID_finder

# Install dependencies
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests with pytest (recommended)
pytest Tests/ -v

# Run specific test suite
pytest Tests/test_green_detection.py -v

# Run with unittest (legacy)
python -m unittest discover Tests
```

### Using Docker

```bash
# Build the Docker image
docker build -t band-id-finder .

# Run tests in container
docker run band-id-finder

# Generate visual artifacts (mount volume to access outputs)
docker run -v $(pwd)/Tests/artifacts:/app/Tests/artifacts band-id-finder python Tests/test_artifacts.py
```

## 📁 Project Structure

```
Band-ID_finder/
├── src/                                    # Source code
│   ├── green_detector.py                  # Main detection interface
│   ├── image_processors.py                # Image processing utilities
│   ├── find_green.py                      # Green blob finder
│   ├── find_red.py                        # Red blob finder
│   ├── find_green_vid.py                  # Video processing
│   └── Band-Tracker_VIDEO_CLIP_TEST.py    # Video tracking script
│
├── Tests/                                  # Test suite
│   ├── test_green_detection.py            # Unit tests for detection
│   ├── test_image_processors.py           # Unit tests for processors
│   ├── test_data_generator.py             # Synthetic test data generator
│   ├── test_artifacts.py                  # Visual artifact generator
│   ├── artifacts/                         # Generated test outputs
│   └── README.md                          # Test documentation
│
├── .github/workflows/                      # CI/CD pipelines
│   ├── docker-image.yml                   # Docker build & test
│   ├── codeql.yml                         # Security scanning
│   └── python-app.yml                     # Python testing
│
├── Dockerfile                              # Container configuration
├── requirements.txt                        # Python dependencies
├── TESTING_GUIDE.md                       # Comprehensive testing docs
└── README.md                              # This file
```

## 🧪 Testing

This project includes a comprehensive test harness with synthetic data generation.

### Test Suites

1. **`test_green_detection.py`** - Core detection tests (15 tests)
   - Single image detection at various positions
   - Small/medium/large blob sizes
   - Video tracking (horizontal, vertical, circular motion)
   - Accuracy validation (<5px error threshold)

2. **`test_image_processors.py`** - Image processing tests (5 tests)
   - Grayscale conversion
   - Blur operations
   - Color space detection
   - Color filtering

3. **`test_artifacts.py`** - Visual output generation (5 tests)
   - Detection sample grids
   - Trajectory plots with error analysis
   - Annotated videos with bounding boxes
   - Performance summary charts

### Running Tests Locally

```bash
# Run all tests
pytest Tests/ -v

# Run with coverage
pytest Tests/ --cov=src --cov-report=html

# Generate visual artifacts
python Tests/test_artifacts.py -v
# Outputs saved to: Tests/artifacts/<timestamp>_<commit>/
```

### Test Artifacts

After running `test_artifacts.py`, view generated outputs:
- `detection_samples.png` - 3x3 grid of detections
- `trajectory_plot.png` - Position over time + error graphs
- `circular_trajectory.png` - 2D path visualization
- `accuracy_summary.png` - Performance metrics
- `input_video.mp4` - Original test video
- `output_video_annotated.mp4` - With detection overlays

All artifacts are tagged with git commit ID for version tracking.

## 🔧 Usage

### Basic Detection

```python
from src.green_detector import GreenBlobDetector
import cv2

# Initialize detector
detector = GreenBlobDetector()

# Load image
image = cv2.imread('path/to/image.jpg')

# Detect green blob
detection = detector.detect_in_image(image)

if detection:
    print(f"Center: {detection['center']}")
    print(f"Bounding box: {detection['bbox']}")
    print(f"Area: {detection['area']}")

    # Visualize
    vis_image = detector.visualize_detection(image, detection)
    cv2.imshow('Detection', vis_image)
    cv2.waitKey(0)
```

### Video Processing

```python
from src.green_detector import GreenBlobDetector

detector = GreenBlobDetector()

# Process entire video
detections = detector.detect_in_video('path/to/video.mp4')

# Analyze trajectory
for i, detection in enumerate(detections):
    if detection:
        print(f"Frame {i}: Center at {detection['center']}")
```

## 🔄 Algorithm Swapping

The test suite is algorithm-agnostic. To test a new detection method:

1. **Modify** `src/green_detector.py::detect_in_image()`
2. **Maintain the interface**: Return `{'bbox', 'center', 'area', 'contour'}`
3. **Run tests**: `pytest Tests/test_green_detection.py -v`
4. **Compare visually**: `python Tests/test_artifacts.py -v`

See `TESTING_GUIDE.md` for detailed instructions.

## 📊 Performance Metrics

Current detector achieves:
- ✅ **100% detection rate** on synthetic test data
- ✅ **~1.6px mean position error**
- ✅ **<3.2px max error** across all test scenarios
- ✅ **Handles fast motion** without losing tracking

## 🐳 Docker

The Docker image includes:
- Python 3.11-slim base
- OpenCV system dependencies
- All Python packages from requirements.txt
- Runs pytest by default

```bash
# Build
docker build -t band-id-finder .

# Run tests
docker run band-id-finder

# Run specific test
docker run band-id-finder pytest Tests/test_green_detection.py -v

# Interactive shell
docker run -it band-id-finder bash
```

## 🔒 CI/CD

GitHub Actions automatically:
- ✅ Build Docker image on push/PR
- ✅ Run all tests in container
- ✅ Execute CodeQL security scanning (weekly + on changes)
- ✅ Test with native Python environment

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run tests: `pytest Tests/ -v`
4. Generate artifacts: `python Tests/test_artifacts.py -v`
5. Commit with descriptive message
6. Push and create PR

## 📚 Documentation

- **`TESTING_GUIDE.md`** - Comprehensive testing documentation
- **`Tests/README.md`** - Test suite details
- **`Dockerfile`** - Container configuration
- **`.github/workflows/`** - CI/CD pipeline definitions

## 🐛 Troubleshooting

### Import Errors
Ensure you're running from the project root and PYTHONPATH includes `src/`:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Docker Build Fails
If you see apt-get errors, ensure Docker daemon is running:
```bash
docker --version
docker info
```

### Tests Fail in Docker but Pass Locally
Check that all source files are in `src/` directory and copied to the container.

## 📝 Dependencies

- `opencv-python>=4.5` - Computer vision
- `numpy>=1.23` - Array operations
- `pytest>=8.3.4` - Testing framework
- `matplotlib>=3.5.0` - Plotting and visualization
- `importmonkey>=0.3` - Path manipulation

## 📄 License

See LICENSE file for details.

## 👥 Authors

UCSC Research - TTNT Lab

## 🔗 Links

- [GitHub Repository](https://github.com/JennaSeaSC/Band-ID_finder)
- [Testing Guide](TESTING_GUIDE.md)
- [CI/CD Workflows](.github/workflows/)
