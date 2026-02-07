# 📸 Smart Attendance System (Microservices Edition)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-FastAPI-green)
![Frontend](https://img.shields.io/badge/Frontend-Streamlit-red)
![AI Model](https://img.shields.io/badge/AI-MediaPipe-orange)
![Deploy](https://img.shields.io/badge/Docker-Compose-blueviolet)

A robust, **Dockerized** attendance system that uses **Face Recognition** to mark attendance in real-time. This project uses a **Microservices Architecture** to separate the AI/Logic (Backend) from the User Interface (Frontend).

## 🚀 Key Features

* **Microservices Architecture:** Frontend and Backend are completely separated for scalability.
* **No "Heavy" Installation:** Uses **Google MediaPipe** for lightweight, CPU-friendly face recognition (no C++ Build Tools required).
* **Real-Time Recognition:** Instantly identifies students and logs attendance.
* **Anti-Duplicate Logic:** Prevents a student from being marked "Present" multiple times in one day.
* **Docker Ready:** Can be built and run on any machine with a single command.

---

## 🏗️ System Architecture

| Service | Technology | Port | Description |
| :--- | :--- | :--- | :--- |
| **Backend** | FastAPI + SQLite | `8000` | Handles AI processing, Database, and API logic. |
| **Frontend** | Streamlit | `8501` | Provides the UI for Camera input and Dashboard. |

---

## 🛠️ Installation & Usage

You can run this project in **two ways**. Choose the one that suits you.

### Option 1: Using Docker (Recommended) 🐳
*Best for: People who want it to "just work" without installing Python libraries manually.*

1.  **Install Docker Desktop.**
2.  Open your terminal in this project folder.
3.  Run the build command:
    ```bash
    docker-compose up --build
    ```
4.  Open your browser:
    * **App (UI):** [http://localhost:8501](http://localhost:8501)
    * **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option 2: Running Locally (Manual) 💻
*Best for: Development and quick testing if you have Python installed.*

**1. Install Dependencies**
Run this command in the main folder to install libraries for both services:
```bash
pip install -r requirements.txt

```

**2. Start the Backend (API)**
Open a terminal and run:

```bash
python backend/main.py

```

*You should see: `Uvicorn running on http://0.0.0.0:8000*`

**3. Start the Frontend (UI)**
Open a **new** terminal window and run:

```bash
streamlit run frontend/app.py

```

---

## 📖 API Documentation

The Backend comes with auto-generated swagger documentation.
Once running, verify the API at: `http://localhost:8000/docs`

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Health check to see if API is running. |
| `POST` | `/register/` | Uploads a photo & ID to enroll a new student. |
| `POST` | `/recognize/` | Accepts a live image to find and mark attendance. |
| `GET` | `/logs/` | Returns the full list of attendance records. |

---

## 📂 Project Structure

```text
SmartAttendance/
├── docker-compose.yml       # Orchestrates Backend & Frontend together
├── requirements.txt         # All Python dependencies
├── backend/                 # 🧠 THE BRAIN
│   ├── Dockerfile
│   ├── main.py              # FastAPI endpoints
│   ├── core.py              # MediaPipe AI Logic
│   └── database.py          # SQLite management
└── frontend/                # 👁️ THE FACE
    ├── Dockerfile
    ├── app.py               # Streamlit Dashboard

```

## 🤝 Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes.
4. Push to the branch and open a Pull Request.

## 📄 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).