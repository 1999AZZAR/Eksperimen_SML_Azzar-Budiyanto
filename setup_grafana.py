import requests
import json

GRAFANA_URL = "http://admin:admin@localhost:3000"

def setup_grafana():
    # 1. Setup Prometheus Datasource
    print("Setting up Prometheus datasource...")
    ds_payload = {
        "name": "Prometheus",
        "type": "prometheus",
        "url": "http://172.17.0.1:9090", # use docker bridge ip to hit prometheus or host network
        "access": "proxy",
        "isDefault": True
    }
    r = requests.post(f"{GRAFANA_URL}/api/datasources", json=ds_payload)
    if r.status_code == 200:
        print("Datasource created.")
    elif r.status_code == 409:
        print("Datasource already exists.")
    else:
        print(f"Failed to create datasource: {r.status_code} {r.text}")

    # 2. Setup Dashboard named "Heart Disease Monitoring - azzar budiyanto"
    print("Setting up Dashboard...")
    dashboard_payload = {
        "dashboard": {
            "id": None,
            "uid": "heart_disease_azzar",
            "title": "Heart Disease Monitoring - azzar budiyanto",
            "tags": [ "templated" ],
            "timezone": "browser",
            "schemaVersion": 36,
            "version": 0,
            "panels": [
                {
                    "type": "timeseries",
                    "title": "Model Accuracy",
                    "gridPos": { "x": 0, "y": 0, "w": 12, "h": 8 },
                    "targets": [ { "expr": "model_accuracy_current", "refId": "A" } ]
                },
                {
                    "type": "timeseries",
                    "title": "Request Latency",
                    "gridPos": { "x": 12, "y": 0, "w": 12, "h": 8 },
                    "targets": [ { "expr": "rate(http_request_duration_seconds_sum[1m]) / rate(http_request_duration_seconds_count[1m])", "refId": "A" } ]
                },
                {
                    "type": "timeseries",
                    "title": "Total Requests",
                    "gridPos": { "x": 0, "y": 8, "w": 12, "h": 8 },
                    "targets": [ { "expr": "http_requests_total", "refId": "A" } ]
                },
                {
                    "type": "timeseries",
                    "title": "CPU Usage",
                    "gridPos": { "x": 12, "y": 8, "w": 12, "h": 8 },
                    "targets": [ { "expr": "system_cpu_usage_percent", "refId": "A" } ]
                }
            ]
        },
        "overwrite": True
    }
    r = requests.post(f"{GRAFANA_URL}/api/dashboards/db", json=dashboard_payload)
    if r.status_code == 200:
        print("Dashboard created successfully.")
        print(r.json())
    else:
        print(f"Failed to create dashboard: {r.status_code} {r.text}")

if __name__ == "__main__":
    setup_grafana()
