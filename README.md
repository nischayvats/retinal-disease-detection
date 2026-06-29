# Retinal Disease Detection using ResNet50 + Grad-CAM

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red) ![Flask](https://img.shields.io/badge/Flask-Web%20App-green) ![Research](https://img.shields.io/badge/ICDSBS%202026-Paper%20%23908-orange)

## Overview
An end-to-end deep learning system for automated detection of retinal diseases and glaucoma using transfer learning (ResNet50) with Grad-CAM visual explainability. Built as a Flask web application with PDF report generation. Published as peer-reviewed research at ICDSBS 2026.

## Publication
**Paper:** A Multi-modal Deep Learning Framework for Early Detection and Prognosis of Retinal Diseases and Glaucoma  
**Conference:** ICDSBS 2026 — Paper #908  
**Authors:** Nischay Vats, Aasin Sayyad  
**Supervisor:** Dr. R. Yamini  
**Institution:** Dept. of Computing Technologies, SRMIST, Kattankulathur, Chennai – 603203

## Tech Stack
- **Deep Learning:** PyTorch, ResNet50 (Transfer Learning)
- **Explainability:** Grad-CAM heatmaps
- **Web App:** Flask, HTML/CSS (Jinja2 templates)
- **Database:** SQLite
- **Report Generation:** PDF generator
- **Language:** Python 3.8+

## Key Features
- Multi-class retinal disease classification across 8 disease categories
- Grad-CAM heatmaps highlighting clinically relevant retinal regions
- Flask web app — upload retinal image → get prediction + heatmap
- Automated PDF report generation for each prediction
- SQLite database logging all predictions

## Project Structure
retinal-disease-detection/

├── app.py                      # Flask web application

├── model.py                    # ResNet50 model definition

├── gradcam.py                  # Grad-CAM implementation

├── database.py                 # SQLite database handler

├── pdf_generator.py            # PDF report generator

├── retina-and-glocuma.ipynb   # Training notebook

├── retinal_database            # SQLite database

├── requirements.txt            # Dependencies

├── templates/                  # HTML templates

└── samples/                    # Sample retinal images
## How to Run
```bash
# Clone the repo
git clone https://github.com/nischayvats/retinal-disease-detection.git
cd retinal-disease-detection

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
```
Then open your browser at `http://localhost:5000`

## Demo
See `demo.mp4` in the repository for a full walkthrough of the application.

## Results
- High classification accuracy across multiple retinal disease categories
- Grad-CAM heatmaps validated on clinically relevant retinal regions
- Successfully detects: Diabetic Retinopathy, Glaucoma, and 6 other retinal conditions
