from ultralytics import YOLO
import sys
import cv2
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

model_path = './runs/segment/train2/weights/best.pt'
image_path = './test_predict/C1_S1_I13.tiff'

save_img = False

if len(sys.argv) > 1:
    image_path = sys.argv[1]
    if len(sys.argv) > 2:
        save_img = True

img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
final_mask = np.zeros_like(img)
for result in results:
    for j, mask in enumerate(result.masks.data):
        mask = mask.numpy() * 255

        mask = cv2.resize(mask, (W, H))
        mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
        mask = mask.astype('uint8')
        final_mask = cv.bitwise_or(final_mask, mask)

mi = cv.bitwise_and(img, final_mask, mask=None)

if save_img:
    cv.imwrite('outputs/out.png', final_mask)

fig, axes = plt.subplots(1, 3, figsize=(10, 10))
ax = axes.flatten()

ax[0].imshow(img)
ax[0].set_axis_off()
ax[0].set_title("Original Image", fontsize=12)

ax[1].imshow(final_mask)
ax[1].set_axis_off()
title = 'Generated Mask'
ax[1].set_title(title, fontsize=12)

ax[2].imshow(mi)
ax[2].set_axis_off()
ax[2].set_title("image X mask", fontsize=12)

fig.tight_layout()
plt.show()
