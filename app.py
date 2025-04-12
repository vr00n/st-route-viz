import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import pandas as pd

st.set_page_config(page_title="Bus Route Feasibility Classifier", page_icon="🚌")

st.title("🚌 Bus Route Feasibility Predictor")
st.caption("Upload up to 10 route images to check if they are Feasible or Infeasible using a YOLOv8 classifier.")

# Load YOLOv8 Model
@st.cache_resource
def load_model():
    model_path = "best.pt"
    return YOLO(model_path)

model = load_model()

uploaded_files = st.file_uploader("Upload Route Images (Max 10)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 10:
        st.error("Please upload up to 10 images only.")
    else:
        results_list = []

        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            st.image(img, caption=f'Uploaded Image: {uploaded_file.name}', use_container_width=True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                img.save(tmp_file.name)
                image_path = tmp_file.name

            with st.spinner(f'Predicting for {uploaded_file.name}...'):
                results = model.predict(image_path)

            result = results[0]
            top_class_idx = result.probs.top1
            class_names = result.names
            top_class_name = class_names[top_class_idx]
            probs = result.probs.data.tolist()
            confidence = probs[top_class_idx] * 100

            results_list.append({
                "Image": uploaded_file.name,
                "Predicted Class": top_class_name,
                "Confidence %": round(confidence, 2)
            })

            st.markdown(f"### Prediction for `{uploaded_file.name}` : `{top_class_name}` ({confidence:.2f}%)")

        st.markdown("## Summary of Results")

        df = pd.DataFrame(results_list)
        st.dataframe(df, use_container_width=True)

        # Optional Download CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "predictions.csv", "text/csv")

