import numpy as np
import cv2
import matplotlib.pyplot as plt


def calculate_psnr(original_img, processed_img, max_pixel_value):
    mse = np.mean((original_img - processed_img) ** 2)
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr


# Load the original and processed images
original_image = cv2.imread('psnr/mask.png', cv2.IMREAD_GRAYSCALE)
processed_image = cv2.imread('psnr/out_cnn.png', cv2.IMREAD_GRAYSCALE)

# Ensure the images have the same dimensions
original_image = cv2.resize(original_image, (processed_image.shape[1], processed_image.shape[0]))

# Convert images to float for accurate calculations
original_image = original_image.astype(np.float64)
processed_image = processed_image.astype(np.float64)

# Calculate the PSNR
max_pixel_value = 255  # Assuming 8-bit grayscale images
psnr = calculate_psnr(original_image, processed_image, max_pixel_value)
res = f"The PSNR value between the original mask and processed image is: {psnr} dB"

fig, axes = plt.subplots(1, 3, figsize=(10, 10))
ax = axes.flatten()

ax[1].imshow(original_image)
ax[1].set_axis_off()
ax[1].set_title("Original Mask", fontsize=12)

ax[2].imshow(processed_image)
ax[2].set_axis_off()
title = 'Generated Mask'
ax[2].set_title(title, fontsize=12)

ax[0].axis([0, 100, 0, 5])
ax[0].text(3, 4, "CNN", style='italic',
        bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
ax[0].text(3, 1, res, style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

plt.show()


