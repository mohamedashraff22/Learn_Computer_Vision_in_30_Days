import joblib
import cv2
from utils import get_face_landmarks

emotions = ["HAPPY", "SAD", "SURPRISED"]

# Load the model directly using joblib (no need for 'with open')
model = joblib.load("./model")

cap = cv2.VideoCapture(0)  # Use default camera (index 0)

ret, frame = cap.read()

while ret:
    ret, frame = cap.read()

    # Check if frame is valid before proceeding
    if not ret:
        break

    # Get landmarks
    face_landmarks = get_face_landmarks(frame, draw=True)

    # SAFETY CHECK: Only predict if a face is actually detected!
    # If face_landmarks is empty [], the model.predict command will crash.
    if len(face_landmarks) > 0:
        # Predict the emotion
        output = model.predict([face_landmarks])

        # Get the text label (Happy/Sad/etc)
        emotion_prediction = emotions[int(output[0])]

        cv2.putText(
            frame,
            emotion_prediction,
            (10, frame.shape[0] - 10),  # Adjusted Y so text is not cut off
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 0),
            5,
        )
    else:
        # Optional: Show a "No Face" warning if you want
        cv2.putText(
            frame,
            "No Face Detected",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    cv2.imshow("frame", frame)

    # Standard way to exit loop (Press 'q' to quit)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
