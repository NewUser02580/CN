# Perform image segmentation using thresholding, region growing, and watershedimport cv2
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg")

# Check whether image is loaded
if image is None:
    print("Image not found!")
    print("Make sure image.jpg is in the same folder as this Python file.")
    exit()

# Convert BGR image to RGB for displaying with Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. THRESHOLDING

# Threshold value = 127
# Pixel > 127 becomes white (255)
# Pixel <= 127 becomes black (0)

_, threshold_image = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)
# 2. REGION GROWING

height, width = gray.shape

seed_x = width // 2
seed_y = height // 2

# Create an empty image for the region
region_growing = np.zeros_like(gray)

# Get the intensity value of the seed pixel
seed_value = gray[seed_y, seed_x]

# Difference allowed from seed pixel
threshold = 20

# Create a queue for storing pixels
queue = [(seed_y, seed_x)]

# Mark the seed pixel
region_growing[seed_y, seed_x] = 255

# Four neighbouring pixels
neighbors = [
    (-1, 0),   # Up
    (1, 0),    # Down
    (0, -1),   # Left
    (0, 1)     # Right
]

# Region growing process
while queue:

    y, x = queue.pop(0)

    for dy, dx in neighbors:

        ny = y + dy
        nx = x + dx

        # Check image boundary
        if 0 <= ny < height and 0 <= nx < width:

            # Check whether pixel is already selected
            if region_growing[ny, nx] == 0:

                # Check intensity similarity
                if abs(int(gray[ny, nx]) - int(seed_value)) <= threshold:

                    region_growing[ny, nx] = 255

                    queue.append((ny, nx))


# 3. WATERSHED SEGMENTATION

# Convert image to grayscale
gray_ws = gray.copy()

# Threshold the image
_, binary = cv2.threshold(
    gray_ws,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Remove small noise
kernel = np.ones((3, 3), np.uint8)

opening = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=2
)

# Find sure background
sure_background = cv2.dilate(
    opening,
    kernel,
    iterations=3
)

# Find distance from background
distance = cv2.distanceTransform(
    opening,
    cv2.DIST_L2,
    5
)

# Find sure foreground
_, sure_foreground = cv2.threshold(
    distance,
    0.5 * distance.max(),
    255,
    0
)

sure_foreground = np.uint8(sure_foreground)

# Unknown region
unknown = cv2.subtract(
    sure_background,
    sure_foreground
)

# Create markers
num_markers, markers = cv2.connectedComponents(
    sure_foreground
)

# Add 1 so background is not marker 0
markers = markers + 1

# Mark unknown area as 0
markers[unknown == 255] = 0

# Apply Watershed
watershed_image = image.copy()

markers = cv2.watershed(
    watershed_image,
    markers
)

# Watershed boundary is marked as -1
watershed_image[markers == -1] = [0, 0, 255]

# Convert for displaying
watershed_rgb = cv2.cvtColor(
    watershed_image,
    cv2.COLOR_BGR2RGB
)

# DISPLAY ALL RESULTS

plt.figure(figsize=(12, 8))

# Original image
plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

# Thresholding result
plt.subplot(2, 2, 2)
plt.imshow(threshold_image, cmap="gray")
plt.title("Thresholding")
plt.axis("off")

# Region Growing result
plt.subplot(2, 2, 3)
plt.imshow(region_growing, cmap="gray")
plt.title("Region Growing")
plt.axis("off")

# Watershed result
plt.subplot(2, 2, 4)
plt.imshow(watershed_rgb)
plt.title("Watershed")
plt.axis("off")

plt.tight_layout()
plt.show()