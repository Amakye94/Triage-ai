import streamlit as st
import requests

st.title("🧠 AI Triage System")
st.subheader("Capture Eye Image")

# 📷 Camera input
image = st.camera_input("Take a picture of your eye")

if image is not None:
    st.image(image)

    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            response = requests.post(
                "http://127.0.0.1:8000/predict-image",
                files={"file": image.getvalue()}
            )

            result = response.json()

            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"Triage Level: {result['triage_level']}")