
# Band-ID Finder

This project is designed to process videos and identify specific color bands using various image processing techniques.

## Prerequisites

Before running the tests or using the project, make sure you have the following installed:

- **Python 3.7+**: Ensure Python is installed and accessible via your terminal.
- **Required Libraries**: Install the necessary libraries by running:

  ```sh
  pip install -r requirements.txt
  ```

  The `requirements.txt` includes the following dependencies:
  - OpenCV (`opencv-python`)
  - Numpy (`numpy`)
  - ImportMonkey (`importmonkey`)

## Running the Tests

The tests for this project are located in the `Tests` directory. To run the tests, follow these steps:

### 1. Navigate to the Project Root Directory

Open a terminal and change the directory to the project root:

```sh
cd /path/to/Band-ID_finder
```

### 2. Run All Tests Using `unittest`

To run all the tests, use the `unittest` module with the `discover` command. This command will find and run all tests in the `Tests` directory:

```sh
python -m unittest discover Tests
```

### 3. Run a Specific Test File

If you only want to run a specific test file, you can do so by specifying the file directly:

```sh
python Tests/test_image_processors.py
```

## Troubleshooting

- **File Paths**: Ensure that you are running the tests from the project root directory so that relative paths work correctly.

## Example Output

When you run the tests, you should see output similar to the following:

```sh
.....
----------------------------------------------------------------------
Ran 5 tests in 0.004s

OK
```

This indicates that all tests have passed successfully.

## Directory Structure

Below is a simplified version of the project directory structure for reference:

```
Band-ID_finder/
│
├── Band-Tracker_VIDEO_CLIP_TEST.py
├── README.md
├── requirements.txt
├── Tests/
│   ├── test_image_processors.py
│   └── ...
├── Videos/
│   ├── DJI_0003.MP4
│   ├── ...
├── find_green.py
├── find_red.py
├── image_processors.py
├── ...
```

