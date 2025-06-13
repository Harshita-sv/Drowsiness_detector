This is an advanced, real-time Drowsiness Detection System that detects sleepiness and fatigue based on **eye aspect ratio (EAR)**, **mouth aspect ratio (MAR)** for yawning, and **head tilt angle** using facial landmarks. It alerts the user via alarm and saves session reports in JSON.

Built with a combination of OpenCV, dlib, face_recognition, and threading, this project balances accuracy with simplicity and is ideal for real-world demos, internships, or portfolio showcases.

 🚀 Features

- 👀 **Eye Aspect Ratio (EAR)** for eye closure detection
- 😮 **Mouth Aspect Ratio (MAR)** for yawn detection
- 🤯 **Head Tilt Detection** using chin and nose landmark angle
- 🔊 **Plays alarm sound** when drowsiness is detected
- 📈 **Real-time overlay UI** with EAR, MAR, tilt angle, and drowsy/yawn counters
- 📦 **Session data saved** as `.json` report with duration, EAR stats, yawn frequency
- 🎨 **Color-coded alert levels** (Normal → Severe)
- 🧪 **Press `s` to save session manually**, `q` to quit
- 💡 Inspired by [CID - An Education Hub](https://www.youtube.com/@CIDanEducationHub)

---

🧠 What I Added Beyond Tutorials

This project started with inspiration from CID’s YouTube code, but includes many additional enhancements:
- ✅ **Multi-factor alert logic**: EAR + MAR + head pose
- ✅ **JSON session summary export**
- ✅ **Status overlay with dynamic alert level**
- ✅ **Improved facial landmark drawing**
- ✅ **Threaded alarm (non-blocking sound)**
- ✅ **Yawn detection with counters and timestamping**

---

 🛠 Installation Instructions

1️⃣ Clone the repository
2️⃣ Create Environment (Recommended with Anaconda)
3️⃣ Install Dependencies
```bash
conda install -c conda-forge dlib
conda install -c conda-forge face_recognition
pip install opencv-python playsound numpy scipy
```

---
🧪 How to Run

1. Make sure your **webcam is not being used by Snap Camera**
2. Activate your environment:
```bash
conda activate drowsy
```

3. Run the detector:
```bash
python drowsiness.py
```

4. While running:
- Press `q` to quit
- Press `s` to manually save the session

---

## 📁 Sample Output Files

Session logs are automatically saved in JSON format:
```
{
  "date": "2025-06-14 03:42:51",
  "duration_minutes": 0.9451646606127421,
  "drowsy_episodes": 0,
  "total_yawns": 0,
  "average_ear": 0.28624294172097453,
  "min_ear": 0.1412233975251142,
  "yawn_frequency": 0
}

---


## 📬 Connect With Me

**Harshita Joshi**  
📧 harshitavdoshi3104@gmail.com  
🌐 [GitHub](https://github.com/Harshita-sv)

> "Built with curiosity, inspired by YouTube, and enhanced through hands-on learning."_ 🔥
