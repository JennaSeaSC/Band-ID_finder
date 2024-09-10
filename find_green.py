# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

# Load the image
image = cv2.imread('./Grn_DJI_0060.JPG')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# Define a wider range of green color in HSV
lower_green = np.array([30, 40, 40])
upper_green = np.array([90, 255, 255])

# Create a mask for the green color
mask = cv2.inRange(hsv, lower_green, upper_green)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour based on area
largest_contour = max(contours, key=cv2.contourArea)

# Get the bounding rectangle for the largest contour
x, y, w, h = cv2.boundingRect(largest_contour)

# Crop the image using the dimensions of the bounding rectangle
cropped_image = image[y:y+h, x:x+w]

# Bitwise-AND mask and original image to extract green areas
green_parts = cv2.bitwise_and(image, image, mask=mask)

def show_images(show_filtered):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(green_parts)
    ax[1].set_title('Green Filtered Image')

    ax[2].imshow(cropped_image)
    ax[2].set_title('Cropped Image')
    ax[2].axis('off')

    plt.show()

interact(show_images, show_filtered=widgets.Checkbox(value=False, description='Show Green Filter'))
