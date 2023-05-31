import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('patek.jpg', 0)

# Add Gaussian noise to the image
mean = 0
stddev = 25
noisy_image = image + np.random.normal(mean, stddev, image.shape).astype(np.uint8)

# Compute the 2D Fourier Transform of the noisy image
fft_noisy = np.fft.fft2(noisy_image)

# Apply a noise reduction filter in the frequency domain
cutoff_frequency = 20  # Set the cutoff frequency value
rows, cols = image.shape
center_row, center_col = rows // 2, cols // 2
fft_noisy[center_row - cutoff_frequency:center_row + cutoff_frequency,
          center_col - cutoff_frequency:center_col + cutoff_frequency] = 0

# Perform inverse Fourier Transform
denoised_image = np.fft.ifft2(fft_noisy)

# Convert complex values to magnitude
denoised_image = np.abs(denoised_image)

# Display the denoised image
plt.imshow(denoised_image, cmap='gray')
plt.title('Denoised Image')
plt.axis('off')
plt.show()
