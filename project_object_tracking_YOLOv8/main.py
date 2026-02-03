import cv2
from ultralytics import YOLO

# load yolov8 model
model = YOLO("yolov8n.pt")

# load video
video_path = "./testing_video/test_video.mp4"
cap = cv2.VideoCapture(video_path)

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:
        # detect & track objects
        results = model.track(frame, persist=True)  # persist to keep IDs across frames

        # plot results
        frame_ = results[0].plot()

        # visualize
        cv2.imshow("frame", frame_)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
