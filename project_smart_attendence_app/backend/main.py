from fastapi import FastAPI, File, UploadFile, Form
import numpy as np
import cv2
import uvicorn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import database as db
from core import FaceSystem

app = FastAPI()
db.init_db()
face_sys = FaceSystem()


@app.get("/")
def read_root():
    return {"message": "MediaPipe Attendance API Running"}


@app.post("/register/")
async def register_student(
    file: UploadFile = File(...), student_id: str = Form(...), name: str = Form(...)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if not db.add_student(student_id, name):
        return {"status": "error", "message": "ID already exists"}

    success, msg = face_sys.register_face(img, student_id)
    if success:
        return {"status": "success", "message": f"Student {name} registered!"}
    return {"status": "error", "message": msg}


@app.post("/recognize/")
async def recognize_student(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    student_id, score = face_sys.recognize(img)

    if student_id != "Unknown":
        success, msg = db.log_attendance(student_id)
        return {
            "id": student_id,
            "score": score,
            "attendance": "marked" if success else "already_present",
            "message": msg,
        }

    return {"id": "Unknown", "score": score, "attendance": "none"}


@app.get("/logs/")
def get_logs():
    return db.get_logs()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
