# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

# Load the image
image = cv2.imread('./Red_DJI_0030.JPG')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

# Define a red color range in RGB
lower_red = np.array([100, 0, 0])
upper_red = np.array([255, 30, 30])

# Create a mask for the red color using RGB limits
mask = cv2.inRange(image, lower_red, upper_red)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Check if any contour is found
if contours:
    # Find the largest contour based on area
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate moments for the largest contour
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        # Calculate x, y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Draw a larger circle around the centroid
        marked_image = cv2.circle(image.copy(), (cX, cY), radius=30, color=(0, 255, 0), thickness=5)
    else:
        marked_image = image.copy()
else:
    marked_image = image.copy()

# Get the bounding rectangle for the largest contour (if any contour is found)
if contours:
    x, y, w, h = cv2.boundingRect(largest_contour)
    # Crop the image using the dimensions of the bounding rectangle
    cropped_image = image[y:y+h, x:x+w]
else:
    cropped_image = image.copy()  # Default to original if no contours

# Bitwise-AND mask and original image to extract red areas
red_parts = cv2.bitwise_and(image, image, mask=mask)

def show_images(show_filtered):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(marked_image)
    ax[0].set_title('Original Image with Marked Center')
    ax[0].axis('off')

    ax[1].imshow(red_parts)
    ax[1].set_title('Red Filtered Image')

    ax[2].imshow(cropped_image)
    ax[2].set_title('Cropped Image')
    ax[2].axis('off')

    plt.show()

interact(show_images, show_filtered=widgets.Checkbox(value=False, description='Show Red Filter'))
