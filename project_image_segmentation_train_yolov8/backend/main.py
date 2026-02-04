# backend/main.py
# 🐱 FastAPI Backend for Cat Segmentation

import os
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
from fastapi.responses import Response
import io
import cv2

app = FastAPI(title="🐱 Cat Segmentation API")

# Hardcoded Model Path
MODEL_PATH = "models/best.pt"

if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    # 🛠️ FORCE FIX: Manually set the class name so it says "Cat" not "item"
    model.model.names = {0: "Cat"}
    print("✅ Model loaded successfully!")
else:
    model = None
    print(f"❌ Error: Model not found at {MODEL_PATH}")


@app.post("/segment")
async def segment_image(image_file: UploadFile = File(...)):
    if model is None:
        return Response(content="Model file not found.", status_code=500)

    # 1. Read Image
    image_bytes = await image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 2. Inference
    # 🛠️ TWEAK: Lowered confidence to 0.15 to catch 'uncertain' cats
    results = model(image, conf=0.15)

    # 3. Plot Result
    res_plotted = results[0].plot()
    res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    res_image = Image.fromarray(res_plotted)

    # 4. Return as PNG
    img_byte_arr = io.BytesIO()
    res_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return Response(content=img_byte_arr.getvalue(), media_type="image/png")


# run with: uvicorn backend.main:app --reload
