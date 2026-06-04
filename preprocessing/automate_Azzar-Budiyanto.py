"""
automate_Azzar-Budiyanto.py
Automated preprocessing pipeline for Heart Disease Dataset.
Mengkonversi langkah preprocessing dari eksperimen ke dalam bentuk otomatis.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib


def load_data(raw_dir='heart_disease_raw'):
    print('[1/6] Memuat dataset...')
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    df = pd.read_csv(os.path.join(raw_dir, 'heart_disease.csv'), names=columns, na_values="?")
    print(f'      Total data: {len(df)} sampel')
    return df


def clean_data(df):
    print('[2/6] Membersihkan data...')
    # Handle missing values (marked as '?' in the dataset)
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f'      Baris dengan missing values dihapus: {before - after}')
    
    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f'      Duplikat dihapus: {before - after} baris')
    print(f'      Sisa data: {after} sampel')
    
    # Convert target to binary (0 = no disease, 1 = disease)
    # The original target is 0-4
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    print(f'      Target converted to binary: {df["target"].value_counts().to_dict()}')
    
    return df


def encode_features(df):
    print('[3/6] Encoding fitur kategorikal...')
    # Heart disease dataset has some categorical features encoded as numbers, 
    # but 'ca' and 'thal' might need careful handling if they were objects.
    # Since we used na_values="?", they should be numeric now.
    
    # No explicit categorical encoding needed for this dataset version as it's mostly numeric
    # but we can ensure types are correct.
    for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
        df[col] = pd.to_numeric(df[col])
        
    return df


def split_and_scale(df, test_size=0.2, val_size=0.2, random_state=42):
    print('[4/6] Split dan normalisasi data...')
    X = df.drop('target', axis=1).values
    y = df['target'].values

    # First split: train and temp (test + val)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
    )
    
    # Second split: temp into test and val
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f'      Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}')
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


def save_preprocessed(X_train, X_val, X_test, y_train, y_val, y_test, scaler, df, output_dir='heart_disease_preprocessing'):
    print('[5/6] Menyimpan data preprocessed...')
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    df.to_csv(os.path.join(output_dir, 'heart_disease_cleaned.csv'), index=False)

    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f'      {f}: {size:,} bytes')


def preprocess(raw_dir='heart_disease_raw', output_dir='heart_disease_preprocessing'):
    print('=' * 50)
    print('AUTOMATED PREPROCESSING PIPELINE: HEART DISEASE')
    print('=' * 50)

    df = load_data(raw_dir)
    df = clean_data(df)
    df = encode_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(df)
    save_preprocessed(X_train, X_val, X_test, y_train, y_val, y_test, scaler, df, output_dir)

    print('=' * 50)
    print('Preprocessing selesai!')
    print('=' * 50)
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    preprocess()
