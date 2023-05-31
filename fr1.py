import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = plt.imread('patek.jpg')

# Convert the image to grayscale if needed
if len(image.shape) > 2:
    image = np.mean(image, axis=2)

# Compute the 2D Fourier Transform
fft_image = np.fft.fft2(image)

# Shift the zero-frequency component to the center
shifted_fft_image = np.fft.fftshift(fft_image)

# Compute the magnitude spectrum for visualization
magnitude_spectrum = np.abs(shifted_fft_image)

# Plot the original image and its magnitude spectrum
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.log(1 + magnitude_spectrum), cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')

plt.tight_layout()
plt.show()
