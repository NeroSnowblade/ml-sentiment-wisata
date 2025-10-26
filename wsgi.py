import os
# import initialization helpers and the Flask app
from app import app, init_db, load_or_train_model, seed_db_if_empty, STATIC_DIR, WC_DIR

# ensure static and wordcloud dirs exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(WC_DIR, exist_ok=True)

# init database and model at import time so the WSGI server can use it
init_db()
# attach model to app module's MODEL variable
app.MODEL = load_or_train_model()
seed_db_if_empty()

# Expose `app` for WSGI servers (waitress-serve will import this module and use `app`)

# Optional: when running locally via `python wsgi.py` serve with waitress
if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=int(os.environ.get('PORT', '5000')))
