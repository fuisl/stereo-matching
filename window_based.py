import functools
import cv2
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
def window_based(
    img1, img2, disparity_range=16, kernel_size=5, saving=True, cost_name="l1"
):
    """Window-based comparison of two images"""

    if cost_name == "l1":
        diff_func = np.abs
    elif cost_name == "l2":
        diff_func = lambda x: np.square(x)

    # Check if the images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")

    # Initialize the difference image
    depth = np.zeros(img1.shape, dtype=np.uint8)
    scale = 3
    max_value = 255 if cost_name == "l1" else 255 ** 2

    kernel_half = int((kernel_size - 1) / 2)

    for y in range(kernel_half, img1.shape[0] - kernel_half):
        for x in range(kernel_half, img1.shape[1] - kernel_half):
            min_diff = max_value
            best_disparity = 0

            for d in range(disparity_range):
                total = 0
                value = 0

                for v in range(-kernel_half, kernel_half + 1):
                    for u in range(-kernel_half, kernel_half + 1):

                        value = max_value
                        if x + u - d >= 0:
                            value = diff_func(
                                img1[y + v, x + u] - img2[y + v, x + u - d]
                            )

                        total += value

                if total < min_diff:
                    min_diff = total
                    best_disparity = d

            depth[y, x] = best_disparity * scale

    if saving:
        print("Saving result...")
        cv2.imwrite(f"./results/windows_based_{cost_name}.png", depth)
        cv2.imwrite(
            f"./results/windows_based_{cost_name}_color.png",
            cv2.applyColorMap(depth, cv2.COLORMAP_JET),
        )
        print("Done.")

    return depth


window_based_l1 = functools.partial(window_based, cost_name="l1")
window_based_l2 = functools.partial(window_based, cost_name="l2")

if __name__ == "__main__":
    left_image = "./data/Aloe/Aloe_left_1.png"
    right_image = "./data/Aloe/Aloe_right_1.png"
    disparity_range = 64
    kernel_size = 3

    window_based_l1(left_image, right_image, disparity_range, kernel_size)
    window_based_l2(left_image, right_image, disparity_range, kernel_size)
