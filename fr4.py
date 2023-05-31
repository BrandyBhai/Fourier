import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('patek.jpg', 0)

# Compute the 2D Fourier Transform
fft_image = np.fft.fft2(image)

# Apply a low-pass filter
cutoff_frequency = 20  # Set the cutoff frequency value
rows, cols = image.shape
center_row, center_col = rows // 2, cols // 2
fft_image[center_row - cutoff_frequency:center_row + cutoff_frequency,
          center_col - cutoff_frequency:center_col + cutoff_frequency] = 0

# Perform inverse Fourier Transform
filtered_image = np.fft.ifft2(fft_image)

# Convert complex values to magnitude
filtered_image = np.abs(filtered_image)

# Display the filtered image
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')
plt.show()
