import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('patek.jpg', 0)

# Compute the 2D Fourier Transform
fft_image = np.fft.fft2(image)

# Apply image sharpening
sharpening_factor = 1.5  # Set the sharpening factor
rows, cols = image.shape
center_row, center_col = rows // 2, cols // 2
sharpening_kernel = np.zeros((rows, cols))
sharpening_kernel[center_row - 1:center_row + 2, center_col - 1:center_col + 2] = [[0, -1, 0], [-1, sharpening_factor + 4, -1], [0, -1, 0]]
fft_sharpened = fft_image * sharpening_kernel

# Perform inverse Fourier Transform
sharpened_image = np.fft.ifft2(fft_sharpened)

# Convert complex values to magnitude
sharpened_image = np.abs(sharpened_image)

# Display the sharpened image
plt.imshow(sharpened_image, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')
plt.show()
