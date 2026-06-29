from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import io
import base64
from datetime import datetime
import sqlite3
import hashlib
import json
import os

app = Flask(__name__)
app.secret_key = 'retinal_disease_detection_2026_secret_key'

from model import ResNet50MultiLabel
from database import init_db, create_user, verify_user, save_patient_record, get_user_patients
from gradcam import generate_gradcam
from pdf_generator import generate_pdf_report

device = torch.device('cpu')
model = ResNet50MultiLabel(num_classes=8, pretrained=False)
checkpoint = torch.load('models/retinal_disease_model_complete.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class_names = ['Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
class_codes = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user_id = verify_user(username, password)
        if user_id:
            session['user_id'] = user_id
            session['username'] = username
            return redirect(url_for('upload'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        success = create_user(username, password, email)
        if success:
            return redirect(url_for('login'))
        else:
            return render_template('signup.html', error='Username already exists')
    
    return render_template('signup.html')

@app.route('/upload')
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('upload.html', username=session['username'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    patient_name = request.form['patient_name']
    patient_age = request.form['patient_age']
    patient_gender = request.form['patient_gender']
    patient_contact = request.form['patient_contact']
    image_file = request.files['image']
    
    image = Image.open(image_file).convert('RGB')
    image_np = np.array(image)
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
    
    predictions = (probabilities >= 0.5).astype(int)
    detected_diseases = [class_names[i] for i in range(8) if predictions[i] == 1]
    confidence_dict = {class_names[i]: float(probabilities[i]) for i in range(8)}
    
    if len(detected_diseases) == 0:
        detected_diseases = ['Normal']
    
    risk_level = 'Low'
    if any(probabilities[[2, 4]] > 0.7):
        risk_level = 'High'
    elif any(probabilities[[1, 3, 5, 6, 7]] > 0.6):
        risk_level = 'Medium'
    
    gradcam_image = generate_gradcam(model, image_tensor, image_np)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_filename = f"patient_{timestamp}.jpg"
    gradcam_filename = f"gradcam_{timestamp}.jpg"
    
    image_path = os.path.join('uploads', image_filename)
    gradcam_path = os.path.join('uploads', gradcam_filename)
    
    os.makedirs('uploads', exist_ok=True)
    image.save(image_path)
    cv2.imwrite(gradcam_path, cv2.cvtColor(gradcam_image, cv2.COLOR_RGB2BGR))
    
    patient_id = save_patient_record(
        session['user_id'],
        patient_name,
        patient_age,
        patient_gender,
        patient_contact,
        image_path,
        json.dumps(detected_diseases),
        json.dumps(confidence_dict),
        risk_level
    )
    
    session['last_prediction'] = {
        'patient_id': patient_id,
        'patient_name': patient_name,
        'patient_age': patient_age,
        'patient_gender': patient_gender,
        'detected_diseases': detected_diseases,
        'confidence_scores': confidence_dict,
        'risk_level': risk_level,
        'gradcam_filename': gradcam_filename,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return redirect(url_for('results'))

@app.route('/results')
def results():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if 'last_prediction' not in session:
        return redirect(url_for('upload'))
    
    return render_template('results.html', data=session['last_prediction'])

@app.route('/download_report')
def download_report():
    if 'user_id' not in session or 'last_prediction' not in session:
        return redirect(url_for('login'))
    
    pdf_buffer = generate_pdf_report(session['last_prediction'])
    
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"report_{session['last_prediction']['patient_name']}_{datetime.now().strftime('%Y%m%d')}.pdf"
    )

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join('uploads', filename))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)