import cv2
import functools
import numpy as np


def read_and_convert(func):
    @functools.wraps(func)
    def wrapper(image_path1, image_path2, *args, **kwargs):
        """Read and convert images to grayscale"""
        # Read the images
        img1 = cv2.imread(image_path1, 0)
        img2 = cv2.imread(image_path2, 0)

        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        return func(img1, img2, *args, **kwargs)

    return wrapper


@read_and_convert
def pixel_wise(img1, img2, disparity_range=16, saving=True, cost_name="l1"):
    """Pixel-wise comparison of two images"""

    if cost_name == "l1":
        diff_func = np.abs
    elif cost_name == "l2":
        diff_func = lambda x: np.square(x)

    # Check if the images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")

    # Initialize the difference image
    depth = np.zeros(img1.shape, dtype=np.uint8)
    scale = 16
    max_diff = 255

    # Iterate over the pixels
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            disparity = 0
            cost_min = np.inf
            for k in range(disparity_range):
                cost = max_diff if j < k else diff_func(img1[i, j] - img2[i, j - k])
                if cost < cost_min:
                    cost_min = cost
                    disparity = k

            depth[i, j] = disparity * scale  # Visualize the disparity

    if saving:
        print("Saving result...")
        cv2.imwrite(f"./results/pixel_wise_{cost_name}.png", depth)
        cv2.imwrite(
            f"./results/pixel_wise_{cost_name}_color.png", cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        )
        print("Done.")

    return depth

pixel_wise_l1 = functools.partial(pixel_wise, cost_name="l1")
pixel_wise_l2 = functools.partial(pixel_wise, cost_name="l2")

if __name__ == "__main__":
    left = "data/tsukuba/left.png"
    right = "data/tsukuba/right.png"

    pixel_wise_l1(left, right)
    pixel_wise_l2(left, right)