# Submission Dicoding: Membangun Sistem Machine Learning (Thyroid Disease)

Repositori ini berisi implementasi sistem *machine learning end-to-end* untuk klasifikasi penyakit tiroid (Thyroid Disease) sebagai syarat kelulusan submission Dicoding. Proyek ini mendemonstrasikan integrasi MLOps mencakup prapemrosesan data otomatis, pelacakan eksperimen (*experiment tracking*), CI/CD, serta *serving* model dengan *monitoring* metrik *real-time*.

## Arsitektur Sistem

Proyek ini dibangun dengan komponen berikut:
- **MLflow:** Melacak parameter, metrik, dan artefak model (Random Forest). Format `MLproject` digunakan untuk standarisasi eksekusi.
- **GitHub Actions:** CI/CD otomatis (*training* dan *build* Docker) menggunakan `mlflow run --env-manager=local`.
- **Flask:** Aplikasi *serving* yang menyediakan *endpoint* REST API (`/predict`) untuk inferensi.
- **Prometheus & Grafana:** Sistem *monitoring* untuk memantau metrik performa (*latency*, *throughput*, iterasi error) dan status sistem (*CPU usage*, *RAM usage*) saat *serving* berjalan.

## Struktur Direktori

- `.github/workflows/`: Konfigurasi GitHub Actions untuk CI.
- `thyroid_raw/`: Data mentah penyakit tiroid (Sick dataset).
- `preprocessing/`: Folder utama prapemrosesan sesuai ketentuan Kriteria 1.
    - `Eksperimen_Azzar-Budiyanto.ipynb`: Notebook eksperimen.
    - `automate_Azzar-Budiyanto.py`: Skrip otomasi prapemrosesan.
    - `thyroid_preprocessing/`: Artefak data bersih dan *scaler*.
- `MLProject/`: Definisi proyek MLflow (`MLproject`, `conda.yaml`, `main.py`).

## Panduan Menjalankan Sistem (Lokal)

### 1. Prapemrosesan Data
Jalankan skrip berikut untuk membersihkan data mentah:
```bash
python preprocessing/automate_Azzar-Budiyanto.py
```

### 2. Pelatihan Model (MLflow)
Untuk menjalankan melalui spesifikasi proyek MLflow:
```bash
mlflow run MLProject --experiment-name "Thyroid Disease CI" --env-manager=local
```

### 3. Serving & Monitoring
- **Terminal 1 (Flask & Exporter):**
  ```bash
  python "Monitoring dan Logging/3.prometheus_exporter.py"
  ```
- **Terminal 2 (Prometheus & Grafana via Docker):**
  ```bash
  docker run -d --name prometheus -p 9090:9090 -v "$(pwd)/Monitoring dan Logging/2.prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus:latest
  docker run -d --name grafana -p 3000:3000 grafana/grafana:latest
  ```

## Dataset
Dataset Thyroid Disease (Sick) dari UCI Machine Learning Repository digunakan untuk klasifikasi biner antara "sick" dan "negative". Nilai yang hilang (`?`) telah dihapus dan fitur kategorikal telah di-encode secara otomatis.
