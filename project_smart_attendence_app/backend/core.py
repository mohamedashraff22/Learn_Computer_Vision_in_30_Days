import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import os
import requests
import cv2

# Paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.pkl")
MODEL_PATH = os.path.join(DATA_DIR, "mobilenet_v3_large.tflite")

# ✅ NEW WORKING URL (Google's MobileNet V3 Large)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_large/float32/1/mobilenet_v3_large.tflite"


class FaceSystem:
    def __init__(self):
        print("--- INITIALIZING FACE SYSTEM ---")
        self.setup_directories()
        self.download_model()

        # 1. Face Detector
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

        print(f"Loading model from: {MODEL_PATH}")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model failed to download. Check your internet connection."
            )

        # READ FILE INTO MEMORY
        with open(MODEL_PATH, "rb") as f:
            model_bytes = f.read()

        # Check for corruption
        if len(model_bytes) < 1000:
            raise ValueError(
                f"❌ Model file is too small ({len(model_bytes)} bytes). It is corrupted. Delete the 'data' folder and try again."
            )

        print(f"Model loaded: {len(model_bytes)} bytes")

        base_options = python.BaseOptions(model_asset_buffer=model_bytes)
        options = vision.ImageEmbedderOptions(base_options=base_options, quantize=True)

        try:
            self.embedder = vision.ImageEmbedder.create_from_options(options)
            print("✅ MediaPipe ImageEmbedder created successfully!")
        except Exception as e:
            print(f"❌ CRITICAL ERROR: {e}")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            raise e

        self.known_embeddings = {}
        self.load_embeddings()

    def setup_directories(self):
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        if not os.path.exists(IMAGES_DIR):
            os.makedirs(IMAGES_DIR)

    def download_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"Downloading MediaPipe Model from {MODEL_URL}...")
            try:
                response = requests.get(MODEL_URL)
                if response.status_code == 200:
                    with open(MODEL_PATH, "wb") as f:
                        f.write(response.content)
                    print("Model Downloaded Successfully!")
                else:
                    print(f"❌ Download Failed. HTTP Status: {response.status_code}")
            except Exception as e:
                print(f"❌ Download Failed: {e}")
                # Clean up empty file
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)

    def load_embeddings(self):
        if os.path.exists(EMBEDDINGS_PATH):
            with open(EMBEDDINGS_PATH, "rb") as f:
                self.known_embeddings = pickle.load(f)

    def save_embeddings(self):
        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(self.known_embeddings, f)

    def save_student_image(self, image_np, student_id):
        file_path = os.path.join(IMAGES_DIR, f"{student_id}.jpg")
        cv2.imwrite(file_path, image_np)

    def extract_face_crop(self, image_np, detection):
        h, w, _ = image_np.shape
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * w)
        y = int(bboxC.ymin * h)
        width = int(bboxC.width * w)
        height = int(bboxC.height * h)

        x = max(0, x)
        y = max(0, y)
        width = min(w - x, width)
        height = min(h - y, height)

        return image_np[y : y + height, x : x + width]

    def get_embedding(self, face_crop):
        try:
            image_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            embedding_result = self.embedder.embed(mp_image)
            if embedding_result.embeddings:
                return embedding_result.embeddings[0].embedding
            return None
        except Exception as e:
            return None

    def register_face(self, image_np, student_id):
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = self.detector.process(image_rgb)

        if not results.detections:
            return False, "No face detected"

        face_crop = self.extract_face_crop(image_np, results.detections[0])
        embedding = self.get_embedding(face_crop)

        if embedding is not None:
            self.known_embeddings[student_id] = embedding
            self.save_embeddings()
            self.save_student_image(image_np, student_id)
            return True, "Success"
        return False, "Could not extract features"

    def recognize(self, image_np):
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = self.detector.process(image_rgb)

        if not results.detections:
            return "Unknown", 0.0

        detection = results.detections[0]
        face_crop = self.extract_face_crop(image_np, detection)
        curr_emb = self.get_embedding(face_crop)

        if curr_emb is None or not self.known_embeddings:
            return "Unknown", 0.0

        keys = list(self.known_embeddings.keys())
        values = np.array(list(self.known_embeddings.values()))

        scores = np.dot(values, curr_emb)
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        if best_score > 0.92:
            return keys[best_idx], float(best_score)

        return "Unknown", float(best_score)
