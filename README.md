# 🌱 Plant Disease Detection

A Deep Learning based web application that detects plant diseases from leaf images.

## 🚀 Features

- Upload plant leaf image
- Preview selected image
- Detect plant disease using Deep Learning
- Display prediction confidence
- FastAPI backend
- Responsive frontend

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- FastAPI
- HTML
- CSS
- JavaScript

## 📁 Project Structure

```text
Plant-Disease-Detection/
├── backend/
├── frontend/
├── dataset/
├── model/
├── main.py
├── train_model.py
├── requirements.txt
└── README.md
## ⚙️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Taniyabansal/PLANT-DISEASE-DETECTION.git
cd PLANT-DISEASE-DETECTION
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
### 6. Open the frontend

Open `frontend/index.html` in your browser.
### 7. Detect Disease

- Click **Choose File**
- Select a plant leaf image
- Click **Detect Disease**
- View the predicted disease and confidence score