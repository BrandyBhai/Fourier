import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('patek.jpg', 0)

# Compute the 2D Fourier Transform
fft_image = np.fft.fft2(image)

# Apply Gaussian blur
sigma = 5  # Set the standard deviation for the Gaussian kernel
rows, cols = image.shape
center_row, center_col = rows // 2, cols // 2

# Create a Gaussian kernel of the same size as the input image
x = np.linspace(0, cols - 1, cols)
y = np.linspace(0, rows - 1, rows)
x, y = np.meshgrid(x, y)
gaussian_kernel = np.exp(-((x - center_col) ** 2 + (y - center_row) ** 2) / (2 * sigma ** 2))
gaussian_kernel /= np.sum(gaussian_kernel)

# Apply the Gaussian kernel in the frequency domain
fft_blurred = fft_image * gaussian_kernel

# Perform inverse Fourier Transform
blurred_image = np.fft.ifft2(fft_blurred)

# Convert complex values to magnitude
blurred_image = np.abs(blurred_image)

# Display the blurred image
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image')
plt.axis('off')
plt.show()
