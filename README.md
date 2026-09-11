# Face Recognition Attendance System  
**Real-time Automated Attendance using Face Recognition (Flask + OpenCV + scikit-learn)**

A beautiful, fully functional web-based attendance system that uses your webcam to detect and recognize faces in real time — no manual entry required!

---

### Features
- Real-time face detection & recognition
- Automatic attendance marking with timestamp
- Add new users with live face capture (10 images)
- Clean, responsive web interface (Bootstrap)
- Daily CSV attendance reports
- Works offline — no internet required
- Robust error handling (OneDrive sync issues fixed)
- Windows webcam compatibility (CAP_DSHOW fix)
- No scikit-learn version warnings
- Press **ESC** to stop attendance session

---

### Tech Stack
- **Python** – Backend
- **Flask** – Web framework
- **OpenCV** – Face detection & webcam handling
- **scikit-learn (KNN)** – Face recognition model
- **Pandas** – CSV attendance management
- **Haar Cascade** – Pre-trained frontal face detector
- **HTML/CSS/Bootstrap** – Frontend UI

---

### Project Structure
```
face-recognition-attendance-system/
│
├── app.py                     Main application (Flask + OpenCV)
├── home.html                  Web interface
├── haarcascade_frontalface_default.xml   Face detection model
├── background.png             Background image for camera window
│
├── Attendance/                Daily CSV files (auto-created)
├── static/
│   ├── faces/                 Stored user face images
│   └── face_recognition_model.pkl   Trained KNN model
│
└── README.md                  This file
```

---

### How to Run (Step-by-Step)

#### 1. Clone the Repository

#### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Requirements
```bash
pip install flask opencv-python numpy scikit-learn pandas joblib
```

#### 4. Run the App
```bash
python app.py
```

#### 5. Open Browser
Go to:[ http://127.0.0.1:5000]


---

### How to Use

#### Add a New User
1. Enter Name (e.g., `akash s`)
2. Enter ID/Roll No (e.g., `1MJ24CS013`)
3. Click **"Add New User"**
4. Look at the camera → 10 photos will be captured automatically
5. Model retrains instantly

#### Take Attendance
1. Click **"Take Attendance"**
2. A camera window opens
3. Look at the camera — your name appears when recognized
4. Attendance is marked automatically
5. Press **ESC** key when done
6. Browser refreshes — your name appears in the list!

---

### Important Notes
- **Do NOT run this project inside OneDrive/Google Drive/Dropbox** → causes file lock errors  
  → Move to `C:\projects\face-attendance\` or similar local folder
- Close **Teams, Zoom, Skype** before taking attendance (they block webcam)
- Good lighting = better recognition
- Look straight at camera, no glasses/hat for best results

---

### Sample Attendance CSV (`Attendance/Attendance-11_27_25.csv`)
```csv
Name,Roll,Time
Akash S,1MJ24CS013,21:38:35

```
