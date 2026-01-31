import os
import cv2
import mediapipe as mp
from utils import load_image, display_image, save_image

# =============read image=============

image_path = r"D:\me\Mohamed\youtube courses\Learn_Computer_Vision_in_30_Days\assets\LauraPalmer.jpg"
output_path = r"D:\me\Mohamed\youtube courses\Learn_Computer_Vision_in_30_Days\project_face_detection_blurring_opencv\output\saved_image.jpg"

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
    print(results.detections)

    # if results.detections:
    #     for detection in results.detections:
    #         # Get bounding box coordinates
    #         bboxC = detection.location_data.relative_bounding_box
    #         ih, iw, _ = img.shape
    #         bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),
