import sqlite3
import hashlib
from datetime import datetime

DB_NAME = 'retinal_database.db'

def init_db():
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, timeout=20)
        c = conn.cursor()
        
        # Enable Write-Ahead Logging for better concurrency handling
        c.execute('PRAGMA journal_mode=WAL')
        
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            patient_name TEXT NOT NULL,
            patient_age INTEGER NOT NULL,
            patient_gender TEXT NOT NULL,
            patient_contact TEXT NOT NULL,
            image_path TEXT NOT NULL,
            predictions TEXT NOT NULL,
            confidence_scores TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        conn.commit()
    finally:
        if conn:
            conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, email):
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, timeout=20)
        c = conn.cursor()
        hashed_password = hash_password(password)
        c.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                  (username, hashed_password, email))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        if conn:
            conn.close()

def verify_user(username, password):
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, timeout=20)
        c = conn.cursor()
        hashed_password = hash_password(password)
        c.execute('SELECT id FROM users WHERE username = ? AND password = ?',
                  (username, hashed_password))
        result = c.fetchone()
        return result[0] if result else None
    finally:
        if conn:
            conn.close()

def save_patient_record(user_id, patient_name, patient_age, patient_gender, 
                        patient_contact, image_path, predictions, confidence_scores, risk_level):
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, timeout=20)
        c = conn.cursor()
        c.execute('''INSERT INTO patients 
                     (user_id, patient_name, patient_age, patient_gender, patient_contact, 
                      image_path, predictions, confidence_scores, risk_level)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (user_id, patient_name, patient_age, patient_gender, patient_contact,
                   image_path, predictions, confidence_scores, risk_level))
        patient_id = c.lastrowid
        conn.commit()
        return patient_id
    finally:
        if conn:
            conn.close()

def get_user_patients(user_id):
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, timeout=20)
        c = conn.cursor()
        c.execute('SELECT * FROM patients WHERE user_id = ? ORDER BY created_at DESC', (user_id,))
        patients = c.fetchall()
        return patients
    finally:
        if conn:
            conn.close()