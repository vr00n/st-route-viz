# 🚌 Bus Route Feasibility Classifier

via https://www.linkedin.com/posts/vr00n_i-spend-a-lot-of-time-thinking-about-how-activity-7317014394519904257-LZBk?rcm=ACoAAACV1_gBK-jOBoypeEeFk6hlfTTqU-YsecY

Poorly designed routes can negatively impact students—especially those who rely on school for meals or have special needs. 
At NYCSBUS, we've developed strong mental models to quickly assess route quality. 
Inspired by Prof. Judy Fan’s research on how humans outperform AI in visual abstraction tasks, we experiment with evaluating school bus routes visually rather than numerically.

His process involves:

* Converting routes into symbolic visual representations (depots, pickups, and schools marked with distinct symbols).
* Using a routing engine to check if buses reach schools on time.
* Labeling each visual route as feasible or not.
* Training a computer vision model (YOLOv8) on 50 labeled images.
* Launching a web app that predicts route feasibility based solely on its visual signature.

We found the experiment promising, though not highly accurate

-- 

This project uses **YOLOv8** to classify school bus route images into two categories: **Feasible** and **Infeasible**. It's packaged as a simple, user-friendly **Streamlit app** that allows uploading up to 10 images at a time, gives predictions, and provides detailed confidence scores.

## Route Generation and Training Code
https://colab.research.google.com/drive/1ER8GhxOXqDOmNRCOe8a-peH6AMsBo54e?usp=sharing

## 🚀 Demo

[Streamlit Cloud Link](https://st-route-viz-h55atcswtxchjuauiatrgt.streamlit.app/)

---

## 📂 Project Structure

```bus-route-feasibility-classifier/
├── app.py # Streamlit app
├── best.pt # Trained YOLOv8 classification model
├── requirements.txt # Python dependencies
├── packages.txt # System-level dependencies (e.g., libGL for OpenCV)
├── README.md # You’re reading it
└── .streamlit/
└── config.toml # (Optional) Streamlit app config
```

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
