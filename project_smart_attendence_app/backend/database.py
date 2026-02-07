import sqlite3
from datetime import datetime
import pandas as pd
import os

# Get the absolute path of the 'backend' folder
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get the 'SmartAttendance' (Root) folder
ROOT_DIR = os.path.dirname(BACKEND_DIR)
# Set data folder in the Root
DATA_DIR = os.path.join(ROOT_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "attendance.db")


def init_db():
    # Create the root data folder if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS students 
                 (id TEXT PRIMARY KEY, name TEXT, created_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS logs 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, 
                  timestamp TEXT, date TEXT, status TEXT)""")
    conn.commit()
    conn.close()


def add_student(student_id, name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO students (id, name, created_at) VALUES (?, ?, ?)",
            (student_id, name, datetime.now().isoformat()),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def log_attendance(student_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    today_str = datetime.now().strftime("%Y-%m-%d")

    c.execute(
        "SELECT * FROM logs WHERE student_id=? AND date=?", (student_id, today_str)
    )
    if c.fetchone():
        conn.close()
        return False, "Already Marked Today"

    now_str = datetime.now().strftime("%H:%M:%S")
    c.execute(
        "INSERT INTO logs (student_id, timestamp, date, status) VALUES (?, ?, ?, ?)",
        (student_id, now_str, today_str, "Present"),
    )
    conn.commit()
    conn.close()
    return True, f"Marked at {now_str}"


def get_logs():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT logs.date, logs.timestamp, students.name, students.id, logs.status 
        FROM logs 
        JOIN students ON logs.student_id = students.id 
        ORDER BY logs.timestamp DESC
    """,
        conn,
    )
    conn.close()
    return df.to_dict(orient="records")
