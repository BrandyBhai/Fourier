import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('patek.jpg', 0)

# Compute the 2D Fourier Transform
fft_image = np.fft.fft2(image)

# Define the scaling factor
scale = 2  # Set the scaling factor

# Apply scaling in the frequency domain
rows, cols = image.shape
center_row, center_col = rows // 2, cols // 2
scaled_image = np.zeros_like(image)
for i in range(rows):
    for j in range(cols):
        x, y = (j - center_col) / scale + center_col, (i - center_row) / scale + center_row
        x1, y1 = int(np.floor(x)), int(np.floor(y))
        x2, y2 = int(np.ceil(x)), int(np.ceil(y))
        if 0 <= x1 < cols and 0 <= y1 < rows and 0 <= x2 < cols and 0 <= y2 < rows:
            scaled_image[i, j] = (image[y1, x1] + image[y1, x2] + image[y2, x1] + image[y2, x2]) / 4

# Compute the 2D Fourier Transform of the scaled image
fft_scaled = np.fft.fft2(scaled_image)

# Display the scaled image
plt.imshow(scaled_image, cmap='gray')
plt.title('Scaled Image')
plt.axis('off')
plt.show()
