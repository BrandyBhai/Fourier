import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('patek.jpg', 0)

# Compute the 2D Fourier Transform
fft_image = np.fft.fft2(image)

# Define the rotation angle
angle = 30  # Set the rotation angle in degrees

# Apply rotation in the frequency domain
rows, cols = image.shape
center_row, center_col = rows // 2, cols // 2
rotation_matrix = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
                            [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]])
rotated_image = np.zeros_like(image)
for i in range(rows):
    for j in range(cols):
        x, y = np.dot(rotation_matrix, np.array([j - center_col, i - center_row])) + np.array([center_col, center_row])
        x, y = int(round(x)), int(round(y))
        if 0 <= x < cols and 0 <= y < rows:
            rotated_image[i, j] = image[y, x]

# Compute the 2D Fourier Transform of the rotated image
fft_rotated = np.fft.fft2(rotated_image)

# Display the rotated image
plt.imshow(rotated_image, cmap='gray')
plt.title('Rotated Image')
plt.axis('off')
plt.show()
