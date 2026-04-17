# 🧠 AI Triage System (Pupillometry-Based)

## 🚀 Overview

This project is an AI-powered triage system that analyzes eye images (pupil response) to estimate physiological stress levels and classify triage urgency.

It combines **computer vision**, **machine learning**, and **web technologies** to simulate a smart triage assistant.

---

## 🎯 Key Features

* 👁️ **Pupil Detection** using OpenCV
* 🧠 **Machine Learning Model** (Random Forest)
* ⚙️ **FastAPI Backend** for predictions
* 🎨 **Streamlit Frontend** with camera input
* 📷 Real-time image capture from browser
* 🔄 End-to-end AI pipeline

---

## 🧱 System Architecture

```
Camera/Image → Pupil Detection → Feature Extraction → ML Model → Triage Output
```

---

## 🛠️ Tech Stack

* Python
* OpenCV
* Scikit-learn
* FastAPI
* Streamlit
* NumPy / Pandas

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository

```
git clone https://github.com/Amakye94/Triage-ai.git
cd Triage-ai
```

---

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

*(or manually install: fastapi, uvicorn, streamlit, opencv-python, scikit-learn)*

---

### 3️⃣ Train the model

```
python main.py
```

---

### 4️⃣ Start backend (API)

```
uvicorn api:app --reload
```

---

### 5️⃣ Start frontend (UI)

```
streamlit run app.py
```

---

### 6️⃣ Use the system

* Open Streamlit in browser
* Capture eye image 📷
* Click **Analyze**
* View triage result

---

## 📊 Example Output

```
Triage Level: Low
```

---

## ⚠️ Disclaimer

This project is a **prototype** and not medically validated.
It is intended for research, learning, and demonstration purposes only.

---

## 🔥 Future Improvements

* 🎥 Real-time video-based pupil tracking
* 🧠 Model trained on real pupillometry data
* 📊 Confidence scoring & explanations
* 🌐 Cloud deployment
* 📱 Mobile-friendly UI

---

## 👨‍💻 Author

**Ebenezer A. Amakye**

---

## ⭐ If you like this project

Give it a star on GitHub ⭐
