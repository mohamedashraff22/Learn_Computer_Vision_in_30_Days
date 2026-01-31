"""Utility functions for image processing using OpenCV."""

import os
import cv2


# utility functions for image loading and saving
def load_image(image_path):
    """Load an image from the specified file path.

    Args:
        image_path (str): The path to the image file.'
    Returns:
        image (numpy.ndarray): The loaded image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"The specified image path does not exist: {image_path}"
        )

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from the path: {image_path}")

    return image


def save_image(image, output_path):
    """Save an image to the specified file path.

    Args:
        image (numpy.ndarray): The image to be saved.
        output_path (str): The path where the image will be saved.
    """
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    success = cv2.imwrite(output_path, image)
    if not success:
        raise ValueError(f"Failed to save image to the path: {output_path}")


def display_image(window_name, image):
    """Display an image in a window.

    Args:
        window_name (str): The name of the window.
        image (numpy.ndarray): The image to be displayed.
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
