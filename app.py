import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import pandas as pd

st.set_page_config(page_title="Bus Route Feasibility Classifier", page_icon="🚌")

# Title
st.title("🚌 Bus Route Feasibility Predictor")
st.caption("Upload a route image to check if it's Feasible or Infeasible using a YOLOv8 classifier.")

# Load YOLOv8 Model
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Your trained YOLOv8 Classification model
    return YOLO(model_path)

model = load_model()

# File Uploader
uploaded_file = st.file_uploader("Upload Route Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Save to a temp file for YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        img.save(tmp_file.name)
        image_path = tmp_file.name

    # Predict
    with st.spinner('Predicting...'):
        results = model.predict(image_path)

    # Extract Prediction Details
    result = results[0]
    top_class_idx = result.probs.top1
    class_names = result.names
    top_class_name = class_names[top_class_idx]
    probs = result.probs.data.tolist()

    # Verbose Output
    st.markdown("## Prediction Results")
    st.markdown(f"### Predicted Class: `{top_class_name}`")

    # Display Confidence per Class
    df = pd.DataFrame({
        'Class': class_names.values(),
        'Confidence %': [round(p * 100, 2) for p in probs]
    }).sort_values(by="Confidence %", ascending=False)

    st.dataframe(df, use_container_width=True)

    # Final Friendly Verdict
    if top_class_name.lower() == "feasible":
        st.success("🚀 Verdict: This route is Feasible!")
    else:
        st.error("⚠️ Verdict: This route is Infeasible!")

    # Optional Debug info
    with st.expander("🔍 Raw YOLO Output (For Debugging)"):
        st.json(result.probs.json())

