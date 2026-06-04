"""
7.Inference.py
Client simulation for Heart Disease Model serving.
Mengirimkan request ke endpoint /predict dan menampilkan hasil prediksi.
"""

import requests
import json
import time
import numpy as np
import os
import joblib

def main():
    url = "http://localhost:8000/predict"
    
    # Load some test data to simulate real requests
    data_dir = '../Membangun_model/heart_disease_preprocessing'
    if not os.path.exists(data_dir):
        data_dir = 'Membangun_model/heart_disease_preprocessing'
        
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    print(f"Starting simulation. Sending {len(X_test)} requests to {url}...")
    
    # In a real scenario, this would be triggered by user actions.
    # Here we simulate it with a loop.
    for i in range(min(20, len(X_test))):
        sample = X_test[i].tolist()
        payload = {"features": sample}
        
        try:
            start = time.time()
            response = requests.post(url, json=payload)
            latency = time.time() - start
            
            if response.status_code == 200:
                result = response.json()
                print(f"[{i+1}] Actual: {y_test[i]} | Pred: {result['prediction']} | Latency: {latency:.4f}s")
            else:
                print(f"[{i+1}] Error: {response.text}")
                
        except Exception as e:
            print(f"[{i+1}] Connection failed: {e}")
            print("Make sure 3.prometheus_exporter.py is running on port 8000!")
            break
            
        time.sleep(0.5)  # Simulate some delay between requests

    print("\nSimulation finished.")

if __name__ == '__main__':
    main()
