import numpy as np
import cv2
from PIL import Image


# helper function to get the HSV color limits for a given BGR color
def get_limits(color):
    # Convert the input color (list) into a numpy array pretending to be a 1x1 pixel image
    # (OpenCV's cvtColor requires an array, not a simple list)
    c = np.uint8([[color]])

    # Convert that single pixel from BGR to HSV color space
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    # Set Lower Limit: Hue - 10, Saturation=100, Value=100
    # (We take a range of +/- 10 for Hue to catch variations of the color)
    lowerLimit = hsvC[0][0][0] - 10, 100, 100

    # Set Upper Limit: Hue + 10, Saturation=255, Value=255
    # (We set Saturation and Value to max to catch vivid and bright versions of the color)
    upperLimit = hsvC[0][0][0] + 10, 255, 255

    # Convert the limits into unsigned 8-bit integer numpy arrays -> unsigned 0-255, signed -128 to 127
    # (This specific data type is required for OpenCV's inRange function)
    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit


blue = [255, 0, 0]  # BGR format for blue color

cap = cv2.VideoCapture(0)

while True:
    ret, frame = (
        cap.read()
    )  # Capture frame-by-frame , ret is a boolean indicating if the frame was captured successfully

    hsvImage = cv2.cvtColor(
        frame, cv2.COLOR_BGR2HSV
    )  # Convert the captured frame from BGR to HSV color space

    lowerLimit, upperLimit = get_limits(blue)  # Get the HSV color limits for blue color

    mask = cv2.inRange(
        hsvImage, lowerLimit, upperLimit
    )  # Create a binary mask where blue colors are white and others are black

    mask_ = Image.fromarray(mask)  # Convert the mask to a PIL Image

    # It scans the entire image and calculates the smallest rectangle that encloses all the white pixels found in the mask. it returns the coordinates of this rectangle as a tuple (left, upper, right, lower).
    bbox = (
        mask_.getbbox()
    )  # Get the bounding box of the white areas in the mask (rom pilow library)

    # print("Bounding Box:", bbox)  # Print the bounding box coordinates

    # cv2.imshow("frame", mask)

    if bbox is not None:
        # Draw a rectangle around the detected blue object in the original frame
        cv2.rectangle(
            frame,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            (0, 255, 0),
            2,
        )  # Green rectangle with thickness 2

    cv2.imshow(
        "Detected Blue Object", frame
    )  # Show the original frame with the rectangle

    if (
        cv2.waitKey(1) & 0xFF == ord("q")
    ):  # 1 means wait for 1 ms after each frame, &0xFF is a bitwise AND operation to get the last 8 bits
        break

cap.release()
cv2.destroyAllWindows()
