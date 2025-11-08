#!/usr/bin/env python3
"""
Label comments in data/data_komentar.csv with 'positif', 'netral', or 'negatif'.
Creates a backup data_komentar.csv.bak before overwriting.

Usage (PowerShell):
python .\scripts\label_data.py

This script uses a small heuristic Indonesian lexicon. It is not a ML model,
so consider training or using the project's model for better accuracy.
"""
import csv
import re
import os
import shutil

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INP = os.path.join(BASE, 'data', 'data_komentar.csv')
BACKUP = INP + '.bak'
TEMP_OUT = INP + '.tmp'

# Simple heuristic Indonesian lexicon (extend as needed)
POS_WORDS = [
    'bagus','indah','enak','sejuk','keren','mantap','puas','nyaman',
    'seru','asik','asyik','menyenangkan','oke','ok','baik','cantik','hebat','terbaik',
    'suka','senang','recommended','rekomendasi','recom','keren','mantap'
]
NEG_WORDS = [
    'buruk','jelek','kotor','lalat','mahal','overrated','overprice','macet','aneh',
    'sulit','susah','kecewa','sayang','kurang','jorok','payah','parah','bau','menjijikkan',
    'rusak','mengerikan','menyedihkan'
]

WORD_RE = re.compile(r"\w+", re.UNICODE)


def label_rows(input_path=INP):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Backup once
    if not os.path.exists(BACKUP):
        shutil.copy2(input_path, BACKUP)
        print('Created backup:', BACKUP)
    else:
        print('Backup already exists:', BACKUP)

    rows = []
    with open(input_path, 'r', encoding='utf-8', newline='') as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames[:] if reader.fieldnames else ['lokasi', 'review']
        for r in reader:
            rows.append(r)

    if 'label' not in fieldnames:
        fieldnames.append('label')

    counts = {'positif': 0, 'negatif': 0, 'netral': 0}

    for r in rows:
        text = (r.get('review') or '')
        text_low = text.lower()
        words = WORD_RE.findall(text_low)
        pos_count = 0
        neg_count = 0
        for w in words:
            if any(p == w or p in w for p in POS_WORDS):
                pos_count += 1
            if any(n == w or n in w for n in NEG_WORDS):
                neg_count += 1
        if pos_count > neg_count and pos_count >= 1:
            label = 'positif'
        elif neg_count > pos_count and neg_count >= 1:
            label = 'negatif'
        else:
            label = 'netral'
        r['label'] = label
        counts[label] += 1

    # Write to temp file then atomically replace original
    with open(TEMP_OUT, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    shutil.move(TEMP_OUT, input_path)

    print('Updated file:', input_path)
    print('Label counts:')
    for k, v in counts.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    label_rows()
