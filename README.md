# 🚌 Bus Route Feasibility Classifier

via https://www.linkedin.com/posts/vr00n_i-spend-a-lot-of-time-thinking-about-how-activity-7317014394519904257-LZBk?rcm=ACoAAACV1_gBK-jOBoypeEeFk6hlfTTqU-YsecY

Poorly designed routes can negatively impact students—especially those who rely on school for meals or have special needs. 
At NYCSBUS, we've developed strong mental models to quickly assess route quality. 
Inspired by Prof. Judy Fan’s research on how humans outperform AI in visual abstraction tasks, we experiment with evaluating school bus routes visually rather than numerically.

The process involves:

* Converting routes into symbolic visual representations (depots, pickups, and schools marked with distinct symbols).
* Using a routing engine to check if buses reach schools on time.
* Labeling each visual route as feasible or not.
* Training a computer vision model (YOLOv8) on 50 labeled images.
* Launching a web app that predicts route feasibility based solely on its visual signature.

We found the experiment promising, though not highly accurate

---

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

## Prompt Transperancy

## Prompt #1
Create a Python script that reads school bus route information from a CSV file, determines the feasibility of each route against a specific set of rules, and generates a sparse "visual grammar" plot for each route suitable for a machine learning model.
1. Primary Goal: Process Routes from a CSV
The script must read a CSV file using the pandas library.
It should group the data by the route column to process each route individually.
The script should be robust enough to handle potential errors in the CSV data (e.g., missing values, incorrect formats).
2. CSV Schema and Data Extraction Logic
The script must correctly interpret the following columns:
route: The unique identifier for a route.
Seg_No: A non-zero value indicates a pickup stop. A zero value indicates a school drop-off.
pupil_lat, pupil_lon: Coordinates for pickup stops.
school_lat, school_lon: Coordinates for school drop-off stops.
route_first_pickup: The scheduled time for the first pickup (P1) of the route. This value may be in various formats and needs robust parsing.
Sess_Beg: The session begin time for a school. This also needs robust parsing.
Extraction Rules:
Pickups: For each route, extract a list of unique pickup coordinates (pupil_lat, pupil_lon). The order of the pickups (P1, P2, etc.) should be based on their order of appearance in the CSV file for rows with a non-zero Seg_No.
Schools: For each route, extract a list of unique school coordinates (school_lat, school_lon). This list must be sorted based on the Sess_Beg time.
3. Feasibility Check Logic
A function check_route_feasibility must evaluate each route against the following rules. A route is feasible only if it meets all conditions.
Routing Engine: Use the public OSRM demo server (http://router.project-osrm.org) for all driving time calculations. Implement a delay (e.g., ~1.1 seconds) between API calls to avoid rate-limiting.
Depot Location: Assume a fixed depot start location at (40.7281, -73.9406).
Max Pickup Distance: The straight-line (geodesic) distance from the depot to the farthest pickup stop must be ≤ 15 miles.
Max Travel Time: The OSRM-calculated driving time for the path from the first pickup (P1), through all subsequent pickups (P2...Pn), to the first school (S1) must be ≤ 120 minutes.
S1 Arrival Window: The calculated arrival time at the first school (S1) must be between 7:00 AM and 9:00 AM inclusive. The calculation must be: P1 Scheduled Time + 3 min wait at P1 + 1 min wait per subsequent pickup + OSRM P1→S1 travel time.
Route Validity: The script must handle cases where OSRM cannot find a valid route for a required segment (e.g., Depot→P1 or P1→S1). These routes are considered infeasible.
4. Visualization: Sparse Visual Grammar
For each route processed, generate and save a PNG image with a highly specific, minimal style.
OSRM Path: The plot must show the actual driving path from OSRM, not straight lines. This requires fetching the route geometry (overview=full, geometries=geojson). The path should be a thin, semi-transparent grey line.
Specific Symbology:
Depot: Black 'x' marker.
P1 (First Pickup): Large red circle.
Other Pickups (P2+): Small blue circles.
S1 (First School): Large magenta square.
Other Schools (S2+): Small cyan squares.
Minimalist Style: The plot must have no clutter. Remove the title, axis labels, axis ticks, grid, legend, and all plot borders (spines).
Consistent Scale: All plots must have the same fixed coordinate limits (based on a bounding box for Brooklyn) and a fixed aspect ratio (aspect='equal') to ensure all route shapes are comparable.
Output: Save each plot to a subdirectory (route_images_from_csv/). Filenames should include the route ID and its feasibility status (e.g., route_R123_feasible.png).
5. Final Output
In addition to the image files, the script should produce a final summary CSV file (route_feasibility_results.csv) containing the route_id, a boolean is_feasible column, and a column with a list of infeasibility_reasons if applicable.
6. Code & Dependencies
The script should be written in Python.
It will require the following libraries: pandas, requests, geopy, shapely, matplotlib, numpy.

## Prompt#2

I want to build a computer vision app using YOLOv8 to classify bus route images into two categories: "feasible" and "infeasible".

My goals are:

1. **Train a YOLOv8 classifier** using a folder of labeled images (stored in `feasible/` and `infeasible/` directories).

2. **Run this training in Google Colab**, and I want a full Colab notebook that:
   - Installs dependencies
   - Splits the dataset into train/val/test
   - Trains a YOLOv8 classification model
   - Exports the trained model

3. **Build a Streamlit app** that:
   - Lets me upload up to 10 route images
   - Runs the classifier on each one
   - Shows each uploaded image
   - Displays the predicted class ("Feasible" or "Infeasible") and the confidence score
   - Shows a summary table with all predictions
   - Allows downloading the results as CSV

4. I will **host the model and app in a GitHub repo**, so please:
   - Structure the repo cleanly (`app.py`, `requirements.txt`, `packages.txt`, etc.)
   - Include fixes for `libGL.so.1` errors (like adding `packages.txt` with `libgl1`)
   - Make the Streamlit app compatible with Streamlit Cloud (no GUI dependencies like `opencv-python` — use `opencv-python-headless`)

5. I want verbose and friendly output in the app — like prediction tables, raw class probabilities, and human-readable verdicts.

Please generate:
- ✅ Colab training notebook
- ✅ Streamlit `app.py` for batch image classification
- ✅ `requirements.txt` and `packages.txt`
- ✅ GitHub repo structure or ZIP if possible

The app should be fast, clean, and future-proof — ready for deployment or public use.



