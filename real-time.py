import cv2 as cv
from ultralytics import YOLO
import cv2
import numpy as np

s = 0
model_path = './runs/segment/train2/weights/best.pt'
source = cv.VideoCapture(s)

win_name = "Camera Preview"
cv.namedWindow(win_name, cv.WINDOW_NORMAL)

frame_width = int(source.get(3))
frame_height = int(source.get(4))

mode = 0
model = YOLO(model_path)
alive = True
while alive:

    hasFrame, frame = source.read()

    if not hasFrame:
        break

    img = frame
    H, W, _ = img.shape
    results = model(img)
    final_mask = np.zeros_like(img)

    if results:
        for result in results:
            if not result or not result.masks:
                break
            for j, mask in enumerate(result.masks.data):
                mask = mask.numpy() * 255

                mask = cv2.resize(mask, (W, H))
                mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
                mask = mask.astype('uint8')
                final_mask = cv.bitwise_or(final_mask, mask)

    key = cv.waitKey(1)
    if key == 27:
        alive = False
    elif key == ord('p'):
        mode = 0
    elif key == ord('m'):
        mode = 1

    if mode == 1:
        res = final_mask
    else:
        res = frame

    cv.imshow(win_name, res)
source.release()
cv.destroyWindow(win_name)
