import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from preprocess import tokenize_for_ml, normalize_text, generate_wordcloud, STOPWORDS_ID

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'reviews.db')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_rf.pkl')
DATA_DIR = os.path.join(BASE_DIR, 'data')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
WC_DIR = os.path.join(STATIC_DIR, 'wordcloud')

SENTIMENT_LABELS = ['negatif', 'netral', 'positif']

app = Flask(__name__)
app.secret_key = 'secret-key-change-me'  # for flash messages


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        '''CREATE TABLE IF NOT EXISTS reviews (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               text TEXT NOT NULL,
               sentiment TEXT NOT NULL,
               created_at TEXT NOT NULL
           )'''
    )
    # ensure 'location' column exists (safe ALTER TABLE for sqlite)
    cur.execute("PRAGMA table_info(reviews)")
    cols = [r[1] for r in cur.fetchall()]
    if 'location' not in cols:
        try:
            cur.execute("ALTER TABLE reviews ADD COLUMN location TEXT DEFAULT ''")
        except Exception:
            # ignore if cannot alter (older DBs) — best effort
            pass
    # create locations table to store unique location names
    cur.execute(
        '''CREATE TABLE IF NOT EXISTS locations (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               name TEXT NOT NULL UNIQUE
           )'''
    )
    conn.commit()
    conn.close()


def load_or_train_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    # Train a new model from bundled data
    dataset_path = os.path.join(DATA_DIR, 'ulasan_labeled.csv')
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=['ulasan', 'label']).reset_index(drop=True)
    # Build pipeline using our tokenizer
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=tokenize_for_ml,
            preprocessor=None,
            lowercase=False,
            ngram_range=(1, 2),
            min_df=1,
        )),
        ('rf', RandomForestClassifier(n_estimators=300, random_state=42))
    ])
    X = df['ulasan'].tolist()
    y = df['label'].tolist()
    pipe.fit(X, y)
    joblib.dump(pipe, MODEL_PATH)
    return pipe


MODEL = None


def seed_db_if_empty():
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) as c FROM reviews')
    c = cur.fetchone()['c']
    if c == 0:
        # Seed from labeled dataset
        dataset_path = os.path.join(DATA_DIR, 'ulasan_labeled.csv')
        df = pd.read_csv(dataset_path).dropna(subset=['ulasan', 'label'])
        # seed reviews with empty location
        rows = [(row['ulasan'], row['label'], datetime.utcnow().isoformat(), '') for _, row in df.iterrows()]
        cur.executemany('INSERT INTO reviews (text, sentiment, created_at, location) VALUES (?, ?, ?, ?)', rows)
        conn.commit()
    conn.close()


def get_stats():
    conn = get_db()
    cur = conn.cursor()
    counts = {}
    for label in SENTIMENT_LABELS:
        cur.execute('SELECT COUNT(*) as c FROM reviews WHERE sentiment=?', (label,))
        counts[label] = cur.fetchone()['c']
    cur.execute('SELECT COUNT(*) as c FROM reviews')
    total = cur.fetchone()['c']
    # Collect texts per sentiment for wordcloud
    texts = {}
    for label in SENTIMENT_LABELS:
        cur.execute('SELECT text FROM reviews WHERE sentiment=?', (label,))
        texts[label] = [row['text'] for row in cur.fetchall()]
    # location stats: totals per location and per-sentiment counts
    cur.execute(
        """
        SELECT
          COALESCE(location, '') as location,
          COUNT(*) as total,
          SUM(CASE WHEN sentiment='negatif' THEN 1 ELSE 0 END) as negatif,
          SUM(CASE WHEN sentiment='netral' THEN 1 ELSE 0 END) as netral,
          SUM(CASE WHEN sentiment='positif' THEN 1 ELSE 0 END) as positif
        FROM reviews
        GROUP BY COALESCE(location, '')
        ORDER BY total DESC
        """
    )
    location_rows = cur.fetchall()
    location_stats = [dict(r) for r in location_rows]
    conn.close()
    return total, counts, texts, location_stats


@app.route('/')
def dashboard():
    total, counts, texts, location_stats = get_stats()
    # Generate wordclouds
    os.makedirs(WC_DIR, exist_ok=True)
    generate_wordcloud(texts['positif'], STOPWORDS_ID, os.path.join(WC_DIR, 'positif.png'), colormap='Greens')
    generate_wordcloud(texts['negatif'], STOPWORDS_ID, os.path.join(WC_DIR, 'negatif.png'), colormap='Reds')
    generate_wordcloud(texts['netral'], STOPWORDS_ID, os.path.join(WC_DIR, 'netral.png'), colormap='Blues')

    return render_template(
        'dashboard.html',
        total=total,
        counts=counts,
        bar_labels=['Negatif', 'Netral', 'Positif'],
        bar_values=[counts['negatif'], counts['netral'], counts['positif']],
        positif_wc=url_for('static', filename='wordcloud/positif.png'),
        negatif_wc=url_for('static', filename='wordcloud/negatif.png'),
        netral_wc=url_for('static', filename='wordcloud/netral.png'),
        location_stats=location_stats,
    )


@app.route('/input', methods=['GET', 'POST'])
def input_ulasan():
    result = None
    proba = None
    ulasan = ''
    location = ''
    if request.method == 'POST':
        ulasan = request.form.get('ulasan', '').strip()
        location = request.form.get('location', '').strip()
        if not ulasan:
            flash('Silakan masukkan ulasan terlebih dahulu.', 'warning')
            return redirect(url_for('input_ulasan'))
        # Predict
        pred = MODEL.predict([ulasan])[0]
        if hasattr(MODEL, 'predict_proba'):
            try:
                probas = MODEL.predict_proba([ulasan])[0]
                # map class to probability safely
                classes = list(MODEL.classes_)
                proba_map = {classes[i]: float(probas[i]) for i in range(len(classes))}
                proba = proba_map.get(pred, None)
            except Exception:
                proba = None
        result = pred
        # store: ensure location exists in locations table if provided
        conn = get_db()
        cur = conn.cursor()
        if location:
            try:
                cur.execute('INSERT OR IGNORE INTO locations (name) VALUES (?)', (location,))
            except Exception:
                pass
        cur.execute(
            'INSERT INTO reviews (text, sentiment, created_at, location) VALUES (?, ?, ?, ?)',
            (ulasan, result, datetime.utcnow().isoformat(), location or '')
        )
        conn.commit()
        conn.close()
    return render_template('input.html', result=result, proba=proba, ulasan=ulasan, location=location)


@app.route('/api/locations')
def api_locations():
    q = request.args.get('q', '').strip()
    conn = get_db()
    cur = conn.cursor()
    if q:
        cur.execute("SELECT name FROM locations WHERE name LIKE ? ORDER BY name LIMIT 20", (q + '%',))
    else:
        cur.execute("SELECT name FROM locations ORDER BY name LIMIT 50")
    rows = [r['name'] for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)


@app.route('/api/reviews')
def api_reviews():
    # paginated reviews for input page table
    try:
        page = int(request.args.get('page', 1))
    except ValueError:
        page = 1
    try:
        per_page = int(request.args.get('per_page', 10))
    except ValueError:
        per_page = 10
    location = request.args.get('location', '').strip()
    offset = (page - 1) * per_page
    conn = get_db()
    cur = conn.cursor()
    if location:
        cur.execute('SELECT COUNT(*) as c FROM reviews WHERE location=?', (location,))
        total = cur.fetchone()['c']
        cur.execute('SELECT id, text, sentiment, created_at, location FROM reviews WHERE location=? ORDER BY created_at DESC LIMIT ? OFFSET ?', (location, per_page, offset))
    else:
        cur.execute('SELECT COUNT(*) as c FROM reviews')
        total = cur.fetchone()['c']
        cur.execute('SELECT id, text, sentiment, created_at, location FROM reviews ORDER BY created_at DESC LIMIT ? OFFSET ?', (per_page, offset))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    pages = max(1, (total + per_page - 1) // per_page)
    return jsonify({
        'page': page,
        'per_page': per_page,
        'total': total,
        'pages': pages,
        'items': rows,
    })


if __name__ == '__main__':
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(WC_DIR, exist_ok=True)
    init_db()
    MODEL = load_or_train_model()
    seed_db_if_empty()
    # Run app
    app.run(host='0.0.0.0', port=5000, debug=True)
