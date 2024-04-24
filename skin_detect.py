import imageio
import numpy as np
import os

# Loading
result_array = np.load("result_array.npy")
# Create a directory for masked images if it doesn't exist
if not os.path.exists("masked_cropped_images"):
    os.makedirs("masked_cropped_images")

# Specify the folder containing images
image_folder = "cropped_images"

# Get a list of image filenames in the directory
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.jpeg')]

# Threshold for pixel manipulation
threshold = 0.65

# Process each image in the folder
for image_file in image_files:
    # Load image
    image_path = os.path.join(image_folder, image_file)
    try:
        test_image = imageio.imread(image_path)
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found.")
        continue

    height, width, _ = test_image.shape

    # Perform pixel manipulation
    for x in range(height):
        for y in range(width):
            b, g, r = test_image[x, y]
            if abs(result_array[r, g, b]) < threshold:
                test_image[x, y] = [255, 255, 255]

    # Save masked image to new folder
    masked_image_path = os.path.join("masked_cropped_images", f"masked_{image_file}")
    imageio.imwrite(masked_image_path, test_image)

print("All images processed and saved.")
