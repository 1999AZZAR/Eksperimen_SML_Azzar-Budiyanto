# Submission Dicoding: Membangun Sistem Machine Learning

Repositori ini berisi implementasi sistem *machine learning end-to-end* untuk klasifikasi penyakit jantung (Heart Disease) sebagai syarat kelulusan submission Dicoding. Proyek ini mendemonstrasikan integrasi MLOps mencakup prapemrosesan data otomatis, pelacakan eksperimen (*experiment tracking*), CI/CD, serta *serving* model dengan *monitoring* metrik *real-time*.

## Arsitektur Sistem

Proyek ini dibangun dengan komponen berikut:
- **MLflow:** Melacak parameter, metrik, dan artefak model (Random Forest dan Gradient Boosting). Format `MLproject` digunakan untuk standarisasi eksekusi.
- **GitHub Actions:** CI/CD otomatis (*training* dan *build* Docker) menggunakan `mlflow run --env-manager=local`.
- **Flask:** Aplikasi *serving* yang menyediakan *endpoint* REST API (`/predict`) untuk inferensi.
- **Prometheus & Grafana:** Sistem *monitoring* untuk memantau metrik performa (*latency*, *throughput*, iterasi error) dan status sistem (*CPU usage*, *RAM usage*) saat *serving* berjalan.

## Struktur Direktori

- `.github/workflows/`: Konfigurasi GitHub Actions untuk CI.
- `heart_disease_raw/`: Data mentah penyakit jantung (Cleveland).
- `heart_disease_preprocessing/`: Skrip otomasi dan artefak data bersih (di-generate oleh `automate_Azzar-Budiyanto.py`).
- `MLProject/`: Definisi proyek MLflow (`MLproject`, `conda.yaml`, `main.py`).
- `Membangun_model/`: Skrip *training* (`modelling.py`, `modelling_tuning.py`) dan artefak.
- `Monitoring dan Logging/`: Skrip *serving* Flask (`3.prometheus_exporter.py`), klien simulasi (`7.Inference.py` / `infinite_inference.py`), konfigurasi Prometheus, dan bukti tangkapan layar Grafana.

## Panduan Menjalankan Sistem (Lokal)

### 1. Prapemrosesan Data
Jalankan skrip berikut untuk membersihkan data mentah dan menyimpan *array* NumPy beserta *scaler*:
```bash
python automate_Azzar-Budiyanto.py
```

### 2. Pelatihan Model (MLflow)
Jalankan proses *tuning* hiperparameter. Artefak model akan disimpan otomatis.
```bash
cd Membangun_model
python modelling_tuning.py
```

Untuk menjalankan melalui spesifikasi proyek MLflow (seperti yang dieksekusi di CI):
```bash
mlflow run MLProject --experiment-name "Heart Disease CI" --env-manager=local
```

### 3. Serving & Monitoring
Sistem membutuhkan terminal terpisah.
- **Terminal 1 (Flask & Exporter):**
  ```bash
  cd "Monitoring dan Logging"
  python 3.prometheus_exporter.py
  ```
- **Terminal 2 (Simulasi Trafik):**
  ```bash
  cd "Monitoring dan Logging"
  python infinite_inference.py
  ```
- **Terminal 3 (Prometheus & Grafana via Docker):**
  ```bash
  docker run -d --name prometheus -p 9090:9090 -v "$(pwd)/Monitoring dan Logging/2.prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus:latest
  docker run -d --name grafana -p 3000:3000 grafana/grafana:latest
  ```

Metrik dapat diakses via Grafana di `http://localhost:3000` dengan mengonfigurasi *datasource* Prometheus di `http://<IP-Host>:9090`. Dashboard telah dikonfigurasi melalui API pada saat pengujian (`setup_grafana.py`).

## Dataset
Dataset Heart Disease (Cleveland) digunakan dengan modifikasi target klasifikasi biner. Nilai yang bernilai `> 0` diubah menjadi `1` (terindikasi penyakit jantung), sedangkan `0` tetap `0` (normal). Nilai yang hilang (`?`) telah dihapus.
