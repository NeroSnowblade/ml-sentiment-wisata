#!/usr/bin/env python3
"""
Train a RandomForest sentiment classifier and save a scikit-learn Pipeline to disk.

Usage examples:
  python scripts/train_sentiment_model.py \
      --data data/ulasan_labeled.csv \
      --out models/sentiment_rf.pkl

It will:
 - load CSV with columns `ulasan` (text) and `label` (negatif/netral/positif)
 - drop rows with missing text/label
 - train a TfidfVectorizer + RandomForest pipeline
 - evaluate on a held-out test split and print metrics
 - save the trained pipeline with joblib to the `--out` path

The model filename `sentiment_rf.pkl` matches the app's default and can be
placed under the `models/` folder so the Flask app can load it.
"""

import os
import argparse
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from preprocess import tokenize_for_ml, normalize_text


def load_data(path, text_col=None, label_col=None):
    df = pd.read_csv(path)

    # If text/label columns explicitly provided, use them
    if text_col and label_col:
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"Specified columns not found in CSV: {text_col}, {label_col}")
    else:
        # Try common header names first
        if 'ulasan' in df.columns and 'label' in df.columns:
            text_col, label_col = 'ulasan', 'label'
        elif 'review' in df.columns and 'label' in df.columns:
            text_col, label_col = 'review', 'label'
        elif len(df.columns) >= 3:
            # fallback: assume CSV has extra first column (e.g., lokasi), so take 2nd and 3rd cols
            text_col = df.columns[1]
            label_col = df.columns[2]
        else:
            raise ValueError("CSV must contain text and label columns (e.g. 'ulasan'/'review' and 'label'), or include at least 3 columns so the 2nd and 3rd will be used")

    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    return df[text_col].astype(str).tolist(), df[label_col].astype(str).tolist()


def build_pipeline(n_estimators=300, random_state=42):
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize_for_ml, lowercase=False, ngram_range=(1,2), min_df=1)),
        ('rf', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state))
    ])
    return pipe


def main(args):
    X, y = load_data(args.data)
    print(f"Loaded {len(X)} rows from {args.data}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y if len(set(y))>1 else None
    )

    pipe = build_pipeline(n_estimators=args.n_estimators, random_state=args.random_state)
    print("Training model...")
    pipe.fit(X_train, y_train)

    print("Evaluating on test split:")
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipe, args.out)
    print(f"Saved trained pipeline to {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train sentiment RandomForest pipeline')
    parser.add_argument('--data', default='data/data_komentar.csv', help='Path to labeled CSV (ulasan,label)')
    parser.add_argument('--out', default='models/sentiment_rf.pkl', help='Output path for trained model (joblib)')
    parser.add_argument('--test-size', type=float, default=0.15, help='Test split proportion')
    parser.add_argument('--n-estimators', type=int, default=300, help='RandomForest n_estimators')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    main(args)
