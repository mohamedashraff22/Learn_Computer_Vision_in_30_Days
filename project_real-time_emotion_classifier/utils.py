"""
Utility functions for real-time emotion classification using facial landmarks.
Added padding to improve face detection on cropped images.
"""

import cv2
import mediapipe as mp

# Initialize FaceMesh once to optimize performance
mp_face_mesh = mp.solutions.face_mesh  # face mesh means facial landmarks
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def get_face_landmarks(image, draw=False, static_image_mode=True):
    # Convert to RGB
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. Try detecting on the ORIGINAL image first (Best for Webcam)
    results = face_mesh.process(image_input_rgb)

    # 2. If no face found, try adding PADDING (Best for Training Data crops)
    if not results.multi_face_landmarks:
        h, w = image.shape[:2]
        pad_size = int(max(h, w) * 0.4)  # 40% padding
        image_padded = cv2.copyMakeBorder(
            image,
            pad_size,
            pad_size,
            pad_size,
            pad_size,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
        image_padded_rgb = cv2.cvtColor(image_padded, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_padded_rgb)

    image_landmarks = []

    if results.multi_face_landmarks:
        # Draw on the ORIGINAL image if requested (and if face was found there)
        # Note: If detection required padding, we can't draw on original easily,
        # but that mostly happens during data prep where we don't see the screen.
        if draw:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec,
            )

        ls_single_face = results.multi_face_landmarks[0].landmark
        xs_ = []
        ys_ = []
        zs_ = []

        for idx in ls_single_face:
            xs_.append(idx.x)
            ys_.append(idx.y)
            zs_.append(idx.z)

        # 3. NORMALIZE (The fix for "Always Sad")
        # We subtract min (position) AND divide by width (scale)
        # This makes the math identical for Close-ups vs Webcam
        min_x, max_x = min(xs_), max(xs_)
        min_y, max_y = min(ys_), max(ys_)

        face_width = max_x - min_x
        face_height = max_y - min_y

        # Avoid division by zero
        if face_width == 0:
            face_width = 0.001
        if face_height == 0:
            face_height = 0.001

        for j in range(len(xs_)):
            # Normalize X and Y relative to the face size (fixing the different images sizes by dividing by face width/height)
            image_landmarks.append((xs_[j] - min_x) / face_width)
            image_landmarks.append((ys_[j] - min_y) / face_height)
            image_landmarks.append(
                zs_[j]
            )  # Z is usually less critical for this 2D classification

    return image_landmarks
