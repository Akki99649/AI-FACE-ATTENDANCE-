import os
import csv
import shutil
import time
import warnings
from datetime import datetime, date

import cv2
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import InconsistentVersionWarning


# ============================================================
# AI SMART FACE RECOGNITION ATTENDANCE SYSTEM
# ============================================================

app = Flask(__name__)
app.secret_key = "face_attendance_secret_key"


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")
STATIC_DIR = os.path.join(BASE_DIR, "static")
FACES_DIR = os.path.join(STATIC_DIR, "faces")

MODEL_PATH = os.path.join(STATIC_DIR, "face_recognition_model.pkl")

# OpenCV's built-in Haar Cascade
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

NUMBER_OF_IMAGES = 10
FACE_SIZE = (50, 50)

# Lower distance = more similar face
FACE_DISTANCE_THRESHOLD = 5000


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    raise RuntimeError("Could not load OpenCV Haar Cascade.")


# ============================================================
# ATTENDANCE FILE
# ============================================================

def get_today_filename():
    """
    Creates a separate attendance CSV file for each day.
    """

    today = date.today().strftime("%d_%m_%Y")

    return os.path.join(
        ATTENDANCE_DIR,
        f"Attendance-{today}.csv"
    )


def ensure_attendance_file():
    """
    Creates today's attendance file if it doesn't exist.
    """

    filename = get_today_filename()

    if not os.path.exists(filename):

        with open(
            filename,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                ["Name", "Roll", "Time"]
            )


# ============================================================
# GET REGISTERED STUDENTS
# ============================================================

def get_registered_students():

    students = []

    if not os.path.exists(FACES_DIR):
        return students

    for folder_name in os.listdir(FACES_DIR):

        folder_path = os.path.join(
            FACES_DIR,
            folder_name
        )

        if not os.path.isdir(folder_path):
            continue

        # Expected format:
        # Name_Roll
        #
        # Example:
        # Akash_101

        if "_" in folder_name:

            name, roll = folder_name.rsplit("_", 1)

        else:

            name = folder_name
            roll = ""

        students.append({
            "name": name,
            "roll": roll,
            "folder": folder_name
        })

    students.sort(
        key=lambda x: x["name"].lower()
    )

    return students


# ============================================================
# TOTAL REGISTERED STUDENTS
# ============================================================

def get_total_students():

    return len(
        get_registered_students()
    )


# ============================================================
# GET TODAY'S ATTENDANCE
# ============================================================

def get_attendance():

    ensure_attendance_file()

    filename = get_today_filename()

    try:

        dataframe = pd.read_csv(
            filename
        )

        if dataframe.empty:
            return []

        return dataframe.to_dict(
            orient="records"
        )

    except Exception:

        return []


# ============================================================
# TOTAL PRESENT TODAY
# ============================================================

def get_total_present():

    attendance = get_attendance()

    return len(attendance)


# ============================================================
# RESIZE FACE
# ============================================================

def prepare_face(face):

    try:

        resized = cv2.resize(
            face,
            FACE_SIZE
        )

        gray = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2GRAY
        )

        return gray

    except Exception:

        return None


# ============================================================
# TRAIN FACE RECOGNITION MODEL
# ============================================================

def train_model():

    faces = []
    labels = []

    if not os.path.exists(FACES_DIR):
        return False

    for folder_name in os.listdir(FACES_DIR):

        folder_path = os.path.join(
            FACES_DIR,
            folder_name
        )

        if not os.path.isdir(folder_path):
            continue

        image_files = os.listdir(
            folder_path
        )

        for image_name in image_files:

            image_path = os.path.join(
                folder_path,
                image_name
            )

            image = cv2.imread(
                image_path
            )

            if image is None:
                continue

            processed_face = prepare_face(
                image
            )

            if processed_face is None:
                continue

            # Convert 50x50 image into
            # one-dimensional array

            flattened = processed_face.flatten()

            faces.append(flattened)

            labels.append(folder_name)

    if len(faces) == 0:

        print("No face images found for training.")

        return False

    X = np.array(faces)
    y = np.array(labels)

    # KNN face recognition model

    model = KNeighborsClassifier(
        n_neighbors=min(5, len(set(labels))),
        weights="distance"
    )

    model.fit(
        X,
        y
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"Model trained successfully using {len(faces)} images."
    )

    return True


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    try:

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore",
                InconsistentVersionWarning
            )

            model = joblib.load(
                MODEL_PATH
            )

        return model

    except Exception as error:

        print(
            "Could not load model:",
            error
        )

        return None


# ============================================================
# MARK ATTENDANCE
# ============================================================

def mark_attendance(student_folder):

    ensure_attendance_file()

    filename = get_today_filename()

    # Split Name_Roll

    if "_" in student_folder:

        name, roll = student_folder.rsplit(
            "_",
            1
        )

    else:

        name = student_folder
        roll = ""

    # Check existing attendance

    try:

        dataframe = pd.read_csv(
            filename
        )

        if "Roll" in dataframe.columns:

            existing_rolls = (
                dataframe["Roll"]
                .astype(str)
                .tolist()
            )

            if str(roll) in existing_rolls:

                return False

    except Exception:

        pass

    current_time = datetime.now().strftime(
        "%H:%M:%S"
    )

    with open(
        filename,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            name,
            roll,
            current_time
        ])

    print(
        f"Attendance marked: {name} - {roll}"
    )

    return True


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    ensure_attendance_file()

    students = get_registered_students()

    attendance = get_attendance()

    return render_template(
        "home.html",
        students=students,
        attendance=attendance,
        total_students=len(students),
        total_present=len(attendance)
    )


# ============================================================
# ADD / REGISTER NEW STUDENT
# ============================================================

@app.route("/add", methods=["POST"])
def add_student():

    name = request.form.get(
        "name",
        ""
    ).strip()

    roll = request.form.get(
        "roll",
        ""
    ).strip()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not name or not roll:

        flash(
            "Please enter both Name and Roll Number.",
            "danger"
        )

        return redirect(
            url_for("home")
        )

    # Remove unsafe characters

    safe_name = "".join(
        character
        for character in name
        if character.isalnum() or character in " -"
    ).strip()

    safe_roll = "".join(
        character
        for character in roll
        if character.isalnum() or character in "-_"
    )

    if not safe_name or not safe_roll:

        flash(
            "Invalid name or roll number.",
            "danger"
        )

        return redirect(
            url_for("home")
        )

    student_folder_name = (
        f"{safe_name}_{safe_roll}"
    )

    student_folder = os.path.join(
        FACES_DIR,
        student_folder_name
    )

    # --------------------------------------------------------
    # CHECK DUPLICATE STUDENT
    # --------------------------------------------------------

    if os.path.exists(student_folder):

        flash(
            "A student with this Name and Roll Number already exists.",
            "warning"
        )

        return redirect(
            url_for("home")
        )

    os.makedirs(
        student_folder,
        exist_ok=True
    )

    # --------------------------------------------------------
    # OPEN WEBCAM
    # --------------------------------------------------------

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        shutil.rmtree(
            student_folder,
            ignore_errors=True
        )

        flash(
            "Could not open webcam. Check your camera connection.",
            "danger"
        )

        return redirect(
            url_for("home")
        )

    captured_images = 0

    print()
    print("=" * 60)
    print("REGISTERING NEW STUDENT")
    print(f"Name: {safe_name}")
    print(f"Roll: {safe_roll}")
    print("=" * 60)
    print("Look at the camera.")
    print("Press ESC to cancel.")
    print()

    while captured_images < NUMBER_OF_IMAGES:

        success, frame = camera.read()

        if not success:

            print(
                "Could not read webcam frame."
            )

            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        detected_faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(80, 80)
        )

        # Display instructions

        cv2.putText(
            frame,
            f"Images: {captured_images}/{NUMBER_OF_IMAGES}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Look at camera | ESC = Cancel",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        for (
            x,
            y,
            width,
            height
        ) in detected_faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

            face = frame[
                y:y + height,
                x:x + width
            ]

            processed_face = prepare_face(
                face
            )

            if processed_face is None:
                continue

            image_number = captured_images

            image_path = os.path.join(
                student_folder,
                f"{safe_name}_{image_number}.jpg"
            )

            cv2.imwrite(
                image_path,
                processed_face
            )

            captured_images += 1

            print(
                f"Captured image {captured_images}/{NUMBER_OF_IMAGES}"
            )

            # Small delay between captures

            time.sleep(0.3)

            break

        cv2.imshow(
            "Register Student - Face Capture",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            # ESC

            print(
                "Registration cancelled."
            )

            break

    camera.release()

    cv2.destroyAllWindows()

    # --------------------------------------------------------
    # CHECK CAPTURE RESULT
    # --------------------------------------------------------

    if captured_images < NUMBER_OF_IMAGES:

        shutil.rmtree(
            student_folder,
            ignore_errors=True
        )

        flash(
            f"Registration cancelled/incomplete. Only {captured_images} images captured.",
            "warning"
        )

        return redirect(
            url_for("home")
        )

    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    trained = train_model()

    if trained:

        flash(
            f"{safe_name} registered successfully and face model trained.",
            "success"
        )

    else:

        flash(
            "Student images were saved, but model training failed.",
            "danger"
        )

    return redirect(
        url_for("home")
    )


# ============================================================
# START FACE RECOGNITION ATTENDANCE
# ============================================================

@app.route("/start", methods=["POST"])
def start_attendance():

    model = load_model()

    if model is None:

        flash(
            "Face recognition model not found. Please register at least one student first.",
            "warning"
        )

        return redirect(
            url_for("home")
        )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        flash(
            "Could not open webcam.",
            "danger"
        )

        return redirect(
            url_for("home")
        )

    print()
    print("=" * 60)
    print("FACE RECOGNITION ATTENDANCE STARTED")
    print("=" * 60)
    print("Look at the camera.")
    print("Press ESC to stop.")
    print()

    recognized_today = set()

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "Could not read webcam."
            )

            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        detected_faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (
            x,
            y,
            width,
            height
        ) in detected_faces:

            face = frame[
                y:y + height,
                x:x + width
            ]

            processed_face = prepare_face(
                face
            )

            if processed_face is None:
                continue

            flattened = processed_face.flatten()

            sample = flattened.reshape(
                1,
                -1
            )

            try:

                prediction = model.predict(
                    sample
                )[0]

                distances, _ = model.kneighbors(
                    sample,
                    n_neighbors=1
                )

                distance = float(
                    distances[0][0]
                )

            except Exception as error:

                print(
                    "Prediction error:",
                    error
                )

                continue

            # ------------------------------------------------
            # FACE MATCH
            # ------------------------------------------------

            if distance < FACE_DISTANCE_THRESHOLD:

                student_folder = str(
                    prediction
                )

                if student_folder not in recognized_today:

                    marked = mark_attendance(
                        student_folder
                    )

                    recognized_today.add(
                        student_folder
                    )

                else:

                    marked = False

                # Extract display name

                if "_" in student_folder:

                    display_name, display_roll = (
                        student_folder.rsplit(
                            "_",
                            1
                        )
                    )

                else:

                    display_name = student_folder
                    display_roll = ""

                label = (
                    f"{display_name} | {display_roll}"
                )

                color = (
                    0,
                    255,
                    0
                )

            else:

                label = "Unknown Face"

                color = (
                    0,
                    0,
                    255
                )

            # ------------------------------------------------
            # DRAW FACE BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                color,
                2
            )

            cv2.rectangle(
                frame,
                (x, y + height - 35),
                (x + width, y + height),
                color,
                -1
            )

            cv2.putText(
                frame,
                label,
                (x + 5, y + height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2
            )

            # Display confidence/distance

            cv2.putText(
                frame,
                f"Distance: {distance:.0f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "AI FACE ATTENDANCE",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press ESC to stop",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "AI Smart Face Recognition Attendance",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            # ESC

            break

    camera.release()

    cv2.destroyAllWindows()

    flash(
        "Face recognition attendance session completed.",
        "success"
    )

    return redirect(
        url_for("home")
    )


# ============================================================
# DELETE STUDENT
# ============================================================

@app.route(
    "/delete/<path:student_folder>",
    methods=["POST"]
)
def delete_student(student_folder):

    # Prevent path traversal

    safe_folder = os.path.basename(
        student_folder
    )

    student_path = os.path.join(
        FACES_DIR,
        safe_folder
    )

    if not os.path.isdir(student_path):

        flash(
            "Student not found.",
            "danger"
        )

        return redirect(
            url_for("home")
        )

    try:

        shutil.rmtree(
            student_path
        )

        # Retrain model after deletion

        remaining_students = get_registered_students()

        if len(remaining_students) > 0:

            train_model()

        else:

            if os.path.exists(MODEL_PATH):

                os.remove(
                    MODEL_PATH
                )

        flash(
            "Student deleted successfully.",
            "success"
        )

    except Exception as error:

        flash(
            f"Could not delete student: {error}",
            "danger"
        )

    return redirect(
        url_for("home")
    )


# ============================================================
# STUDENTS PAGE
# ============================================================

@app.route("/students")
def students_page():

    students = get_registered_students()

    return render_template(
        "home.html",
        students=students,
        attendance=get_attendance(),
        total_students=len(students),
        total_present=get_total_present()
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return {
        "status": "running",
        "registered_students": get_total_students(),
        "model_exists": os.path.exists(MODEL_PATH)
    }


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 65)
    print("        AI SMART FACE RECOGNITION ATTENDANCE")
    print("=" * 65)
    print()
    print("Project folder:")
    print(BASE_DIR)
    print()
    print("Registered students:")
    print(get_total_students())
    print()
    print("Attendance folder:")
    print(ATTENDANCE_DIR)
    print()
    print("Face images folder:")
    print(FACES_DIR)
    print()
    print("Model:")
    print(MODEL_PATH)
    print()
    print("Open this address in your browser:")
    print("http://127.0.0.1:5000")
    print()
    print("Press CTRL+C in this terminal to stop the server.")
    print("=" * 65)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )