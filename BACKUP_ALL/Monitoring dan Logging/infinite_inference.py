import requests
import time
import numpy as np
import os

def main():
    url = "http://localhost:8000/predict"
    data_dir = '../Membangun_model/heart_disease_preprocessing'
    if not os.path.exists(data_dir):
        data_dir = 'Membangun_model/heart_disease_preprocessing'
        
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    print(f"Starting infinite simulation on {url}...")
    
    i = 0
    while True:
        sample = X_test[i % len(X_test)].tolist()
        payload = {"features": sample}
        try:
            requests.post(url, json=payload, timeout=2)
        except Exception:
            pass
        i += 1
        time.sleep(1.0) # 1 request per second

if __name__ == '__main__':
    main()
