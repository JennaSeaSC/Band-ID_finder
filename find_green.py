"find_green_working_tree"
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ipywidgets import interact, widgets
from pathlib import Path

def process_image_and_show(image_path):
    # Load the image
    image = cv2.imread(image_path)
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
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding rectangle for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the image using the dimensions of the bounding rectangle
        cropped_image = image[y:y+h, x:x+w]

        # Bitwise-AND mask and original image to extract green areas
        green_parts = cv2.bitwise_and(image, image, mask=mask)

        # Function to display images
        def show_images(show_filtered):
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(image)
            ax[0].set_title('Original Image')
            ax[0].axis('off')

            ax[1].imshow(green_parts)
            ax[1].set_title('Green Filtered Image')
            ax[1].axis('off')

            ax[2].imshow(cropped_image)
            ax[2].set_title('Cropped Image')
            ax[2].axis('off')

            file_name = os.path.basename(image_path)
            plt.savefig(os.path.join("./output", file_name))
            plt.show(block=False)

        # Use interact to show images with filtering options
        interact(show_images, show_filtered=widgets.Checkbox(value=False, description='Show Green Filter'))

    else:
        print("No green areas detected in the image.")

# Call the function with an example image path
# process_image_and_show("C:/Users/jbcon/Documents/CODE/Band-ID_finder/Green/Grn_DJI_0060.JPG")

directory = Path ("./Band-ID_finder/Green/")
file_names = [f.name for f in directory.iterdir() if f.is_file()]
for file in file_names:
    process_image_and_show(os.path.join(directory, file))
input("Press Enter to stop")
