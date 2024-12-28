import numpy as np


def generate_mask(sample_point, image, tol, w, h):
    mask = np.zeros(image.shape).astype(int)

    red = image[sample_point[0]][sample_point[1]][0]
    green = image[sample_point[0]][sample_point[1]][1]
    blue = image[sample_point[0]][sample_point[1]][2]

    for r in range(sample_point[0] - h, sample_point[0] + h):
        for c in range(sample_point[0] - w, sample_point[0] + w):
            if red - tol < image[r][c][0] < red + tol:
                if green - tol < image[r][c][1] < green + tol:
                    if blue - tol < image[r][c][2] < blue + tol:
                        mask[r][c] = [255, 255, 255]
    return mask


def generate_masked_image(image, mask):
    print(mask.shape)
    mi = np.zeros(mask.shape).astype(int)
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if sum(mask[r][c]) == 255 * 3:
                mi[r][c] = image[r][c]
    return mi
