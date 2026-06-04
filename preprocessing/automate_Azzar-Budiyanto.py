"""
automate_Azzar-Budiyanto.py
Automated preprocessing pipeline for Thyroid Disease (Sick) Dataset.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def preprocess(raw_dir='thyroid_raw', output_dir='preprocessing/thyroid_preprocessing'):
    print('=' * 50)
    print('AUTOMATED PREPROCESSING PIPELINE: THYROID DISEASE')
    print('=' * 50)

    # 1. Load Data
    columns = [
        "age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
        "sick", "pregnant", "pregnant_query", "lithium", "goitre", "tumor", "hypopituitary",
        "psych", "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4",
        "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured", "TBG", "referral_source", "target"
    ]
    
    # Dataset uses '?' for missing and the last col contains target|id
    df = pd.read_csv(os.path.join(raw_dir, 'thyroid.csv'), header=None, na_values='?')
    
    # Fix column assignment (dataset has more cols than my list due to the |id part)
    # We take first 29 columns and name them
    df = df.iloc[:, :27]
    df.columns = [
        "age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
        "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
        "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych",
        "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4",
        "T4U_measured", "T4U", "FTI_measured", "FTI", "target"
    ]

    # Clean target: "negative.|3733" -> "negative"
    df['target'] = df['target'].str.split('.').str[0]
    
    # 2. Handle Missing Values
    # drop rows with missing values for simplicity in this pipeline
    df = df.dropna()
    print(f"Data samples after dropping NaN: {len(df)}")

    # 3. Encoding
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    # 4. Split and Scale
    X = df.drop('target', axis=1).values
    y = df['target'].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 5. Save
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
    print(f"Preprocessing complete. Files saved to {output_dir}")
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':
    preprocess()
