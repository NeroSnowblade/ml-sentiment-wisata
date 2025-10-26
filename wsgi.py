import os
import importlib

# import the app module so we can set module-level MODEL variable
app_module = importlib.import_module('app')
app = app_module.app

# ensure static and wordcloud dirs exist
os.makedirs(app_module.STATIC_DIR, exist_ok=True)
os.makedirs(app_module.WC_DIR, exist_ok=True)

# init database and model at import time so the WSGI server can use it
app_module.init_db()
# set the module-level MODEL variable so routes using MODEL work
app_module.MODEL = app_module.load_or_train_model()
app_module.seed_db_if_empty()

# Expose `app` for WSGI servers (waitress-serve will import this module and use `app`)

# Optional: when running locally via `python wsgi.py` serve with waitress
if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=int(os.environ.get('PORT', '5000')))
