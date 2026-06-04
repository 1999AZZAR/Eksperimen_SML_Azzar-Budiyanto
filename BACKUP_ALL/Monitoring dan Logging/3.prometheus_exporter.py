"""
3.prometheus_exporter.py
Integrated Flask serving with Prometheus monitoring for Heart Disease Model.
Menyediakan endpoint /predict untuk inferensi dan /metrics untuk monitoring.
"""

import os
import time
import joblib
import numpy as np
import psutil
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# --- METRICS DEFINITION ---
# 1. Total Requests (Counter)
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')

# 2. Throughput (Counter - used for rate calculation)
THROUGHPUT = Counter('http_requests_throughput', 'Total number of requests processed')

# 3. Request Latency (Histogram)
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency',
                            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])

# 4. Model Predictions (Counter with Labels for classes)
# This is "observasi nyata" of the model output
PREDICTION_COUNT = Counter('model_predictions_total', 'Total Model Predictions', ['result'])

# 5. Inference Errors (Counter)
INFERENCE_ERRORS = Counter('model_inference_errors_total', 'Total Inference Errors')

# 6. System Metrics (Gauges)
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')

# --- LOAD MODEL AND SCALER ---
# Using absolute or relative path that works in the submission structure
MODEL_PATH = '../Membangun_model/best_rf_model.pkl'
SCALER_PATH = '../Membangun_model/heart_disease_preprocessing/scaler.pkl'

# Fallback for different execution contexts
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'Membangun_model/best_rf_model.pkl'
    SCALER_PATH = 'Membangun_model/heart_disease_preprocessing/scaler.pkl'

print(f"Loading model from {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print(f"Loading scaler from {SCALER_PATH}")
scaler = joblib.load(SCALER_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    THROUGHPUT.inc()
    
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "No features provided"}), 400
        
        # Preprocessing
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Inference
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Update metrics with REAL observational data
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)
        PREDICTION_COUNT.labels(result=str(prediction)).inc()
        
        return jsonify({
            "prediction": int(prediction),
            "probability": probabilities.tolist(),
            "status": "success",
            "latency": duration
        })

    except Exception as e:
        INFERENCE_ERRORS.inc()
        return jsonify({"error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    # Update system metrics on each scrape with real data from psutil
    CPU_USAGE.set(psutil.cpu_percent())
    RAM_USAGE.set(psutil.virtual_memory().percent)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/model-metrics', methods=['GET'])
def model_metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == '__main__':
    # Run on port 8000 as configured in prometheus.yml
    app.run(host='0.0.0.0', port=8000)
