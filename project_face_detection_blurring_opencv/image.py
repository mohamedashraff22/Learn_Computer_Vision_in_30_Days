import os
import cv2
import mediapipe as mp
from utils import load_image, display_image, save_image

# =============read image=============

image_path = r"D:\me\Mohamed\youtube courses\Learn_Computer_Vision_in_30_Days\assets\LauraPalmer.jpg"

img = load_image(image_path)
# display_image("Loaded Image", img)

# =============detect faces=============

mp_face_detection = mp.solutions.face_detection  # create face detection object

with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
) as face_detection:
    # Convert the BGR image to RGB before processing.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    print(
        results.detections
    )  # will give None if no faces detected, if there are faces detected it will give list of detection objects, if there is no faces detected it will give None.

# 1. Safety Check: Only proceed if MediaPipe actually found a face
if results.detections is not None:
    # 2. Loop through every face found (in case there are multiple people)
    for detection in results.detections:
        # 3. Extract the location data from the detection object
        location_data = detection.location_data

        # 4. Get "Relative" Bounding Box
        # "Relative" means numbers are Percentages (0.0 to 1.0), NOT pixels.
        bbox = location_data.relative_bounding_box

        # 5. Unpack the percentages (e.g., xmin might be 0.5, meaning start at 50% width)
        x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

        # 6. Get the REAL size of your image in pixels (Height, Width)
        H, W, _ = img.shape

        # 7. The Math Conversion: Percentage * Total_Pixels = Real_Pixel_Location
        # Example: 0.5 (50%) * 1000px width = Pixel #500
        # We wrap in int() because pixels must be whole numbers (cannot have pixel 500.5)
        x1, y1, w, h = int(x1 * W), int(y1 * H), int(w * W), int(h * H)

        # display face bounding box
        # img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 5)

        # =============blur face=============
        img[y1 : y1 + h, x1 : x1 + w :] = cv2.blur(
            img[y1 : y1 + h, x1 : x1 + w :], (40, 40)
        )  # now we do every thing in the face bounding box

# display and save image
cv2.imshow("Image with Face Detection", img)  # show the whole image
cv2.waitKey(0)
cv2.destroyAllWindows()

# save image
output_path = r"D:\me\Mohamed\youtube courses\Learn_Computer_Vision_in_30_Days\project_face_detection_blurring_opencv\output\Image_with_blured_Face.jpg"
save_image(img, output_path)
