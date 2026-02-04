# frontend/app.py
import streamlit as st
import requests
from PIL import Image
import io

# 🎨 UI CONFIGURATION
st.set_page_config(page_title="Cat Segmenter", page_icon="🐱", layout="wide")

st.title("🐱 AI Cat Segmenter")
st.markdown("### The Auto-Magical Cat Detector")

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.info("👇 **Step 1: Upload a photo**")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        # 🛠️ FIX: Updated deprecated parameter
        st.image(image, caption="Your Photo", use_container_width=True)

with col2:
    st.success("👇 **Step 2: See the result**")

    if uploaded_file:
        if st.button("✨ Detect & Segment Cat", type="primary"):
            with st.spinner("🤖 AI is painting the cat..."):
                try:
                    files = {
                        "image_file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    }

                    response = requests.post(
                        "http://127.0.0.1:8000/segment", files=files
                    )

                    if response.status_code == 200:
                        result_image = Image.open(io.BytesIO(response.content))
                        # 🛠️ FIX: Updated deprecated parameter
                        st.image(
                            result_image, caption="AI Result", use_container_width=True
                        )
                        st.balloons()
                    else:
                        st.error(f"Backend Error: {response.text}")

                except Exception as e:
                    st.error("🚨 Backend not connected!")
                    st.code("Run: uv run fastapi dev backend/main.py")

    else:
        st.write("Waiting for image...")
        st.image("https://placekitten.com/400/300", caption="Example Cat", width=300)

# Run with: uv run streamlit run frontend/app.py
