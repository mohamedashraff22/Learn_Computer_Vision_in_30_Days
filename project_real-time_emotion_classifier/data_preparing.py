"""
Data preparation script for real-time emotion classification using facial landmarks.
"""

import os
import cv2
import numpy as np
from utils import get_face_landmarks

data_dir = "./data"

# 1. Check if directory exists
if not os.path.exists(data_dir):
    print(f"ERROR: The directory {data_dir} does not exist!")
else:
    print(f"Found data directory: {data_dir}")

output = []

# 2. Add prints to track the loop
try:
    for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
        full_emotion_path = os.path.join(data_dir, emotion)

        # Check if it's actually a folder
        if not os.path.isdir(full_emotion_path):
            continue

        print(f"Processing folder: {emotion}")

        for image_path_ in os.listdir(full_emotion_path):
            image_path = os.path.join(data_dir, emotion, image_path_)

            image = cv2.imread(image_path)

            # 3. Check if image loaded correctly
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                continue

            face_landmarks = get_face_landmarks(image)

            # 4. handle if no face is detected (Prevent crashes)
            if not face_landmarks:
                print(f"No face detected in {image_path_}")
                continue

            face_landmarks.append(int(emotion_indx))
            output.append(face_landmarks)

    print(f"Total samples collected: {len(output)}")

    # 5. Only save if we have data
    if len(output) > 0:
        np.savetxt("data.txt", np.asarray(output))
        print("Successfully saved data.txt")
    else:
        print("Output was empty. No file created.")

except Exception as e:
    print(f"An error occurred: {e}")
