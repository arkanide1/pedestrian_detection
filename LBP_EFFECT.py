import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# Load the image

path = 'Resources/Photos/lady.jpg'
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Set parameters for LBP
radius = 1
n_points = 8 * radius

# Apply LBP to the image
lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')

# Display the original and LBP images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap=plt.cm.gray)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(lbp_image, cmap=plt.cm.gray)
plt.title('LBP Image')
plt.axis('off')

plt.show()
