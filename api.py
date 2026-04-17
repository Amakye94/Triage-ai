from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import joblib
from pupil import extract_pupil_features_from_image

# ✅ DEFINE APP FIRST
app = FastAPI()

# ✅ LOAD MODEL
model = joblib.load("triage_model1.pkl")


@app.get("/")
def home():
    return {"message": "Triage AI API is running 🚀"}


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image file"}

        features = extract_pupil_features_from_image(img)

        if features is None:
            return {"error": "No pupil detected"}

        # TEMP mapping
        eda_mean = features["pupil_area"] / 1000
        eda_std = 0.1
        eda_max = eda_mean
        eda_min = eda_mean

        bvp_mean = features["pupil_perimeter"]
        bvp_std = 1
        bvp_max = bvp_mean
        bvp_min = bvp_mean

        data = np.array([[
            eda_mean, eda_std, eda_max, eda_min,
            bvp_mean, bvp_std, bvp_max, bvp_min
        ]])

        prediction = model.predict(data)[0]

        return {
            "triage_level": "Low" if prediction == 0 else "Moderate"
        }

    except Exception as e:
        return {"error": str(e)}