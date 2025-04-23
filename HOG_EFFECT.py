import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# Load the image
path = 'Resources/Photos/lady.jpg'
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Compute HOG features and visualizations
features, hog_image = hog(image, pixels_per_cell=(
    8, 8), cells_per_block=(2, 2), visualize=True)

# Improve the visual representation of HOG
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Display the original image and HOG image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap=plt.cm.gray)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
plt.title('HOG Visualization')
plt.axis('off')

plt.show()
