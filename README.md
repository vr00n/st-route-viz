# 🚌 Bus Route Feasibility Classifier

This project uses **YOLOv8** to classify school bus route images into two categories: **Feasible** and **Infeasible**. It's packaged as a simple, user-friendly **Streamlit app** that allows uploading up to 10 images at a time, gives predictions, and provides detailed confidence scores.

## 🚀 Demo

[Streamlit Cloud Link](https://st-route-viz-h55atcswtxchjuauiatrgt.streamlit.app/)

---

## 📂 Project Structure

``bus-route-feasibility-classifier/
├── app.py # Streamlit app
├── best.pt # Trained YOLOv8 classification model
├── requirements.txt # Python dependencies
├── packages.txt # System-level dependencies (e.g., libGL for OpenCV)
├── README.md # You’re reading it
└── .streamlit/
└── config.toml # (Optional) Streamlit app config``

---

## 🧠 Model Info

- Model type: YOLOv8 Classification
- Input: Static images of bus routes
- Output: "Feasible" or "Infeasible" + confidence %

Trained using `yolov8n-cls.pt` base model on custom-labeled image dataset split into:
- `feasible/`
- `infeasible/`

---

## 🖼️ App Features

✅ Upload up to 10 route images  
✅ View image previews  
✅ Get predictions with confidence scores  
✅ Download results as CSV  
✅ Debug-friendly expandable section for raw probability data  

---

## 🧪 Local Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/bus-route-feasibility-classifier.git
cd bus-route-feasibility-classifier

# Install Python dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
☁️ Streamlit Cloud Deployment
This repo is Streamlit Cloud–ready. It includes:

packages.txt to install missing libGL dependency

Headless OpenCV (opencv-python-headless) to avoid GUI issues

Cached model loading for performance
