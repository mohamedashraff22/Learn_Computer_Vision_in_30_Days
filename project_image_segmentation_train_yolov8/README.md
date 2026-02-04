# 🐱 Custom YOLOv8 Instance Segmentation Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8s--seg-green)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)

<img src="/assets/readmeImages/image_segmentaion_cat.png" alt=".">

An end-to-end computer vision project that trains a custom **YOLOv8 Instance Segmentation** model on the **Open Images V7** dataset (specifically for Cats) and deploys it using a decoupled **FastAPI** backend and **Streamlit** frontend.

## 🌟 Features

* **Automated Data Pipeline:** Uses **FiftyOne** to download specific classes from Open Images V7 and automatically converts binary masks to YOLOv8 Polygon format.
* **Cloud Training:** Jupyter Notebook script designed for **Google Colab** to leverage free GPU compute.
* **Data Visualization:** Built-in scripts to inspect dataset distribution and visualize ground-truth polygon masks.
* **Decoupled Architecture:**
    * **Backend:** FastAPI service that handles model loading and inference.
    * **Frontend:** Streamlit UI for easy user interaction (upload model + image).
* **UV support:** Fast dependency management using `uv`.

## 📂 Project Structure

```bash
├── colab_training/
│   └── Train_YOLOv8_Segmentation.ipynb  # The Notebook to run on Google Colab
├── backend/
│   └── api.py                           # FastAPI inference engine
├── frontend/
│   └── ui.py                            # Streamlit User Interface
├── requirements.txt                     # Standard pip requirements
└── README.md

```

## 🚀 Phase 1: Training (Google Colab)

Since training YOLOv8 requires significant GPU power, we use Google Colab.

1. Open the `colab_training/Train_YOLOv8_Segmentation.ipynb` file in Google Colab.
2. Run the cells sequentially. The script will:

* Install `ultralytics` and `fiftyone`.
* Download the "Cat" segmentation dataset (Train/Val splits).
* Convert Open Images masks to YOLO Polygon format.
* Train the `yolov8n-seg.pt` model for 50 epochs.
* Generate a `best.pt` model file.

3. **Download** the `best.pt` file from the Colab file browser (`runs/segment/train/weights/best.pt`) to your local machine.

## 💻 Phase 2: Local Deployment

This project uses **uv** for blazing fast dependency management.

### Prerequisites

* Python 3.9+
* `uv` installed (or use standard `pip`)

### 1. Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/yolo-segmentation-app.git](https://github.com/yourusername/yolo-segmentation-app.git)
cd yolo-segmentation-app

# Create virtual env and install dependencies via uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install ultralytics fiftyone fastapi uvicorn streamlit python-multipart opencv-python-headless

```

### 2. Run the Backend (FastAPI)

Open a terminal and start the inference server. This handles the heavy ML processing.

```bash
uv run fastapi dev backend/api.py

```

*The API will be live at `http://127.0.0.1:8000*`

### 3. Run the Frontend (Streamlit)

Open a **second** terminal and start the UI.

```bash
uv run streamlit run frontend/ui.py

```

## 🎮 How to Use

1. Open the Streamlit URL provided in the terminal (usually `http://localhost:8501`).
2. **Upload Model:** Drag and drop the `best.pt` file you downloaded from Google Colab.
3. **Upload Image:** Upload any image containing the object (e.g., a Cat) to test.
4. Click **✨ Segment Now**.
5. The app will send the files to the FastAPI backend, process the segmentation, and display the result with the mask overlay.

## 📊 Dataset & Model Details

* **Source:** Open Images Dataset V7
* **Class:** Cat (customizable in the notebook)
* **Model Architecture:** YOLOv8-Nano Segmentation (`yolov8n-seg`)
* **Format:** YOLO Polygon (Normalized `class x1 y1 x2 y2 ...`)

## 🛠️ Built With

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - State-of-the-art Object Detection/Segmentation.
* [FiftyOne](https://voxel51.com/docs/fiftyone/) - Open-source tool for dataset curation and visualization.
* [FastAPI](https://fastapi.tiangolo.com/) - High-performance web framework for APIs.
* [Streamlit](https://streamlit.io/) - Turns data scripts into shareable web apps.

## 📝 License

This project is licensed under the MIT License.

```

```