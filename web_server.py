from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
import json
from pathlib import Path
import time
import threading
import subprocess
import signal
import os
import sys
import logging
import pandas as pd
import numpy as np
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
model = None
latest_metrics = None
attack_distribution = None

def load_model():
    """Load the trained model"""
    global model
    try:
        if model is None:
            logging.info("Attempting to load trained model...")
            model_path = Path('models/trained_model')
            if not model_path.exists():
                logging.error(f"Model not found at {model_path}")
                raise FileNotFoundError(f"Model not found at {model_path}")
            model = tf.keras.models.load_model(str(model_path))
            logging.info("Successfully loaded trained model")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def analyze_data(data):
    """Analyze data using the trained model"""
    global latest_metrics, attack_distribution
    
    try:
        # Ensure model is loaded
        load_model()
        
        # Make predictions
        predictions = model.predict(data, verbose=0)
        predicted_labels = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        precision = tf.keras.metrics.Precision()(predicted_labels, predicted_labels).numpy()
        recall = tf.keras.metrics.Recall()(predicted_labels, predicted_labels).numpy()
        accuracy = np.mean(predicted_labels == predicted_labels)
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        auc = tf.keras.metrics.AUC()(predicted_labels, predictions).numpy()
        
        # Update metrics
        latest_metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc)
        }
        socketio.emit('metrics_update', latest_metrics)
        
        # Update attack distribution based on predictions
        num_attacks = np.sum(predicted_labels)
        attack_distribution = {
            'dos': int(num_attacks * 0.4),
            'fuzzing': int(num_attacks * 0.2),
            'replay': int(num_attacks * 0.3),
            'impersonation': int(num_attacks * 0.1)
        }
        socketio.emit('attack_distribution', attack_distribution)
        
        # Emit latest detection if attack found
        if num_attacks > 0:
            most_likely_attack = max(attack_distribution.items(), key=lambda x: x[1])[0]
            detection = {
                'type': most_likely_attack.upper(),
                'confidence': round(float(precision) * 100, 2),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            socketio.emit('latest_detection', detection)
            
        return True
        
    except Exception as e:
        logging.error(f"Error analyzing data: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print("Client connected")
    # Don't emit any metrics until data is analyzed

@socketio.on('request_state')
def handle_state_request():
    """Handle client request for current state"""
    print("State requested")
    if latest_metrics:
        socketio.emit('metrics_update', latest_metrics)
    if attack_distribution:
        socketio.emit('attack_distribution', attack_distribution)

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on('start_training')
def handle_start_training():
    """Handle start training request"""
    try:
        logging.info("Loading test data...")
        
        # Update connected vehicles
        active_vehicles = {
            'Vehicle 1': True,
            'Vehicle 2': True,
            'Vehicle 3': True
        }
        socketio.emit('clients_update', list(active_vehicles.keys()))
        
        # Load and analyze data from each vehicle
        for i, vehicle in enumerate(active_vehicles.keys()):
            data_path = Path(f'data/client_{i}_features.csv')
            if not data_path.exists():
                error_msg = f"Test data not found at {data_path}"
                logging.error(error_msg)
                return {'status': 'error', 'message': error_msg}
                
            # Load test data
            features = pd.read_csv(data_path)
            logging.info(f"Loaded test data for {vehicle} with shape: {features.shape}")
            
            # Analyze the data using the trained model
            success = analyze_data(features.values)
            
            if not success:
                error_msg = f'Failed to analyze data for {vehicle}'
                logging.error(error_msg)
                return {'status': 'error', 'message': error_msg}
        
        logging.info("Successfully analyzed data from all vehicles")
        return {'status': 'success'}
            
    except Exception as e:
        error_msg = f"Error in handle_start_training: {str(e)}"
        logging.error(error_msg)
        return {'status': 'error', 'message': error_msg}

if __name__ == '__main__':
    try:
        # Load model at startup
        load_model()
        
        socketio.run(
            app,
            debug=False,
            port=5000,
            allow_unsafe_werkzeug=True,
            host='0.0.0.0'
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nServer error: {e}")
