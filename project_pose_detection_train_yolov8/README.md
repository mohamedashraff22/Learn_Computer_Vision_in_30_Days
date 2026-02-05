# 🦴 YOLOv8 Pose Detection: Humans & Dogs

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8--Pose-green)
![Colab](https://img.shields.io/badge/Platform-Google%20Colab-orange)

A comprehensive computer vision project exploring **Pose Estimation** (Keypoint Detection) using YOLOv8. This repository demonstrates two distinct workflows:
1.  **Training** a custom model to detect Dog skeletons (24 keypoints).
2.  **Inference** using a pre-trained Medium model to track Human skeletons in videos.

---

## 📂 Project Structure

| Notebook File | Description | Model Used |
| :--- | :--- | :--- |
| `YOLOv8_dogs_pose_detection_training.ipynb` | **Training Pipeline:** Fine-tunes a model on the `dog-pose` dataset. | `yolov8n-pose.pt` (Nano) |
| `YOLOv8m_human_pose_detection_pretrained_no_training.ipynb` | **Inference Pipeline:** Processes video files to detect human pose. | `yolov8m-pose.pt` (Medium) |

---

## 🐶 Part 1: Dog Pose Training
**Goal:** Teach YOLOv8 to understand the anatomy of a dog, which differs significantly from humans (tail, ear positioning, quadruped posture).

<img src="output_videos/dog_image_output.png.png" alt="dog output" width="800">

### Key Features
* **Automated Dataset Management:** Utilizes Ultralytics' `dog-pose.yaml` which automatically downloads ~6,000 annotated dog images.
* **Custom Keypoints:** Tracks **24 specific body parts** including:
    * Legs (Elbows, Knees, Paws)
    * Tail (Start, End)
    * Ears (Base, Tip)
* **Transfer Learning:** Starts with `yolov8n-pose` (Human) and re-trains the head for Dog features.

### How to Run
1.  Open `YOLOv8_dogs_pose_detection_training.ipynb` in Google Colab.
2.  Run the training cell. The dataset (~300MB) will download automatically.
3.  The trained model (`best.pt`) will be saved for download.

---

## 🏃‍♂️ Part 2: Human Pose Video Inference
**Goal:** High-accuracy skeleton tracking on video footage without any training required.

<img src="output_videos/human_image_output1.png" alt="dog output" width="600"><img src="output_videos/human_image_output2.png" alt="dog output" width="585">

### Key Features
* **Medium Model (`yolov8m-pose`):** We use the "Medium" version instead of "Nano" for superior accuracy on moving targets, trading slightly slower speed for better stability.
* **Video Processing:** Full pipeline to upload a raw video (`.mp4`, `.avi`), process it frame-by-frame, and download the result with skeleton overlays.
* **Zero-Shot Inference:** Uses the pre-trained COCO-Pose weights (trained on 200k+ human images).

### How to Run
1.  Open `YOLOv8m_human_pose_detection_pretrained_no_training.ipynb`.
2.  Upload any video file containing humans.
3.  The script will process the video and automatically trigger a download of the result.

---

## 🛠️ Installation & Requirements

If running locally, you need the following dependencies:

```bash
pip install ultralytics opencv-python matplotlib

```

*Note: For video processing, `ffmpeg` is recommended.*

## 📊 Results

### Dog Skeleton (Training Result)

*(The model successfully identifies the 24 keypoints even on unseen dog breeds.)*

### Human Tracking (Video Result)

*(The Medium model maintains stable tracking even during fast movement.)*

---

## 🧠 Theory: Why different models?

* **For Dogs (Nano):** We used `yolov8n-pose` for training because it is lightweight and faster to fine-tune on a smaller dataset (~6k images) without overfitting.
* **For Humans (Medium):** We used `yolov8m-pose` for inference because video analysis requires higher stability. The "Medium" model has ~25M parameters (vs 3M in Nano), making it much better at handling occlusion and complex movements.

---

## 🤝 Credits

* **Ultralytics YOLOv8:** For the state-of-the-art models and the `dog-pose` dataset curation.
* **Open Images / Stanford Dogs:** Original sources for the dog dataset.
