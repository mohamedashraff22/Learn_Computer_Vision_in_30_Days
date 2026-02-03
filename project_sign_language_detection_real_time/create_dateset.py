import os
import pickle
import mediapipe as mp
import cv2


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = "./asl_dataset"

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  #

        x_ = []
        y_ = []

        path_to_img = os.path.join(DATA_DIR, dir_, img_path)
        img = cv2.imread(path_to_img)

        # Add Padding
        # This adds a black border around the image so the hand isn't touching the edges
        if img is not None:
            h, w, c = img.shape
            # Add a border of 40 pixels (or more if images are large)
            img = cv2.copyMakeBorder(
                img, 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

        if img is None:
            print(f"Skipping bad file: {img_path}")
            continue  # Skip to the next file, don't crash

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

        # Only save if we have exactly 42 features (21 x, 21 y)
        if len(data_aux) == 42:
            data.append(data_aux)
            labels.append(dir_)
        else:
            print(
                f"Skipping image {img_path} due to insufficient landmarks. Found {len(data_aux)} features."
            )


# save the dataset
save_dir = "./models"
save_path = os.path.join(save_dir, "data.pickle")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

f = open(save_path, "wb")
pickle.dump({"data": data, "labels": labels}, f)
f.close()

print(f"Success! Data saved to {save_path}")
