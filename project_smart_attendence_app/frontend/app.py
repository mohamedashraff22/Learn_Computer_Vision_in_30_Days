import streamlit as st
import requests
import pandas as pd
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AI Attendance", layout="wide")
st.title("🎓 Smart Attendance System (MediaPipe)")

menu = st.sidebar.radio(
    "Navigation", ["Take Attendance", "Enroll Student", "View Logs"]
)

if menu == "Take Attendance":
    st.header("📸 Live Attendance")
    img_file = st.camera_input("Snap a photo")

    if img_file:
        files = {"file": ("image.jpg", img_file.getvalue(), "image/jpeg")}
        try:
            with st.spinner("Analyzing..."):
                response = requests.post(f"{API_URL}/recognize/", files=files)

            if response.status_code == 200:
                data = response.json()
                if data["id"] != "Unknown":
                    st.success(f"✅ Verified: {data['id']}")
                    if data["attendance"] == "marked":
                        st.info(f"✨ {data['message']}")
                    else:
                        st.warning(f"⚠️ {data['message']}")
                else:
                    st.error("❌ Person Unknown.")
            else:
                st.error("Server Error")
        except:
            st.error("Cannot connect to backend.")

elif menu == "Enroll Student":
    st.header("📝 New Student")
    with st.form("enroll_form"):
        sid = st.text_input("Student ID")
        name = st.text_input("Name")
        photo = st.file_uploader("Face Photo", type=["jpg", "png", "jpeg"])

        if st.form_submit_button("Register"):
            if sid and name and photo:
                files = {"file": ("image.jpg", photo.getvalue(), "image/jpeg")}
                data = {"student_id": sid, "name": name}
                try:
                    res = requests.post(f"{API_URL}/register/", files=files, data=data)
                    st.write(res.json())
                except:
                    st.error("Backend failed.")

elif menu == "View Logs":
    st.header("📊 History")
    if st.button("Refresh"):
        try:
            res = requests.get(f"{API_URL}/logs/")
            if res.status_code == 200:
                st.dataframe(pd.DataFrame(res.json()), use_container_width=True)
        except:
            st.error("Failed to fetch logs.")
