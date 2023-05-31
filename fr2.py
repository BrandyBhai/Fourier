import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('patek.jpg', 0)

# Compute the 2D Fourier Transform
fft_image = np.fft.fft2(image)

# Perform inverse Fourier Transform
restored_image = np.fft.ifft2(fft_image)

# Convert complex values to magnitude
restored_image = np.abs(restored_image)

# Display the restored image
plt.imshow(restored_image, cmap='gray')
plt.title('Restored Image')
plt.axis('off')
plt.show()
