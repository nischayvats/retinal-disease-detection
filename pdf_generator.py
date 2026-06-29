from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime
import io
import base64

def generate_pdf_report(prediction_data):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "Retinal Disease Detection Report")
    
    c.setFont("Helvetica", 12)
    y_position = height - 100
    
    c.drawString(50, y_position, f"Patient Name: {prediction_data['patient_name']}")
    y_position -= 20
    c.drawString(50, y_position, f"Age: {prediction_data['patient_age']}")
    y_position -= 20
    c.drawString(50, y_position, f"Gender: {prediction_data['patient_gender']}")
    y_position -= 20
    c.drawString(50, y_position, f"Date: {prediction_data['timestamp']}")
    y_position -= 40
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, "Detected Conditions:")
    y_position -= 25
    
    c.setFont("Helvetica", 12)
    for disease in prediction_data['detected_diseases']:
        confidence = prediction_data['confidence_scores'].get(disease, 0)
        c.drawString(70, y_position, f"- {disease}: {confidence*100:.1f}%")
        y_position -= 20
    
    y_position -= 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, f"Risk Level: {prediction_data['risk_level']}")
    
    c.save()
    buffer.seek(0)
    return buffer