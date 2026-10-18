"""ELEC 3111 Tutorial 9: cloud-hostable Python analytics server.

Preserves Tutorial 8 by reading a separate Firebase RTDB path. The ESP32
writes DIRECTLY to Firebase; this server ONLY reads and analyzes.
Uses Node/Python and RTDB concepts from Lectures 5-8. No MySQL/IP fields.
"""
import base64
import csv
import hmac
import io
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

from analysis_python import regression, neural, prepare_rows

BASE = Path(__file__).resolve().parent
PORT = int(os.environ.get('PORT', '10000'))
DEMO_MODE = os.environ.get('DEMO_MODE', '0').lower() in ('1', 'true', 'yes')
DB_URL = os.environ.get('FIREBASE_DATABASE_URL', '').strip()
DB_PATH = os.environ.get('FIREBASE_PATH', 'tutorial9/readings').strip('/')
APP_USER = os.environ.get('APP_USER', '')
APP_PASSWORD = os.environ.get('APP_PASSWORD', '')

def configured():
    return bool(DB_URL and (os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON') or
                              os.environ.get('FIREBASE_SERVICE_ACCOUNT_FILE')))

if not APP_USER or not APP_PASSWORD:
    print('ERROR: Set APP_USER and APP_PASSWORD in the environment.', flush=True)
    sys.exit(1)
if not DEMO_MODE and not configured():
    print('ERROR: Provide Firebase URL and service account credentials, or explicitly set DEMO_MODE=1 for local testing.', flush=True)
    sys.exit(1)

firebase_error = None
ref = None
if not DEMO_MODE:
    try:
        import firebase_admin
        from firebase_admin import credentials, db
        secret_text = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON', '')
        if secret_text:
            cert = credentials.Certificate(json.loads(secret_text))
        else:
            cert = credentials.Certificate(os.environ['FIREBASE_SERVICE_ACCOUNT_FILE'])
        app = firebase_admin.initialize_app(cert, {'databaseURL': DB_URL})
        ref = db.reference(DB_PATH, app=app)
        print('Firebase Admin configured: path=' + DB_PATH, flush=True)
    except Exception as exc:
        print('ERROR: Firebase Admin initialization failed; check credentials and database URL.', flush=True)
        sys.exit(1)


def demo_rows(limit):
    """Deterministic fake readings. Explicitly demo-only; NEVER real Firebase data."""
    end = int(time.time() // 5) * 5000
    out = []
    for j in range(limit):
        i = j + 850
        t = 25.6 + 0.85 * math.sin(i / 14) + 0.1 * math.sin(i / 4)
        h = 51.8 + 5.0 * math.cos(i / 23) + 0.3 * math.sin(i / 3)
        out.append(dict(time=end - (limit - j - 1) * 5000,
                        temperature=round(t, 2), humidity=round(h, 2)))
    return out


def safe_finite(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def get_rows(limit=200):
    limit = min(1000, max(1, int(limit)))
    if DEMO_MODE:
        return demo_rows(limit)
    raw = ref.order_by_child('time').limit_to_last(limit).get()
    if raw is None:
        return []
    if isinstance(raw, dict):
        values = list(raw.values())
    elif isinstance(raw, list):
        values = [x for x in raw if x is not None]
    else:
        raise ValueError('Unexpected Firebase data structure')
    out = []
    for r in values:
        if not isinstance(r, dict):
            continue
        t = safe_finite(r.get('time'))
        temperature = safe_finite(r.get('temperature', r.get('Temperature')))
        humidity = safe_finite(r.get('humidity', r.get('Humidity')))
        if t is None or temperature is None or humidity is None:
            continue
        if -50 <= temperature <= 100 and 0 <= humidity <= 100:
            out.append(dict(time=int(t), temperature=temperature, humidity=humidity))
    return sorted(out, key=lambda x: x['time'])[-limit:]


def analyze(kind, options):
    if not isinstance(options, dict):
        raise ValueError('Options must be an object.')
    if kind not in ('mlr', 'mlp', 'preprocess'):
        raise ValueError('Unknown analysis method')
    rows_count = options.get('rows', 300)
    try:
        rows_count = int(rows_count)
    except (TypeError, ValueError):
        raise ValueError('Rows must be an integer.')
    if not 10 <= rows_count <= 1000:
        raise ValueError('Rows must be 10–1000.')
    rows = get_rows(rows_count)
    if kind == 'mlr':
        return regression(rows, options)
    if kind == 'mlp':
        return neural(rows, options)
    clean, processed, _, mode = prepare_rows(rows, options)
    return dict(method='preprocess', countBefore=len(rows), countAfter=len(processed),
                scale=mode, sample=processed[-5:])


class Handler(BaseHTTPRequestHandler):
    server_version = 'ELEC3111-T9/1.0'

    def auth_ok(self):
        header = self.headers.get('Authorization', '')
        if not header.startswith('Basic '):
            return False
        try:
            decoded = base64.b64decode(header[6:], validate=True).decode('utf-8')
        except (ValueError, UnicodeDecodeError):
            return False
        name, sep, password = decoded.partition(':')
        return bool(sep) and hmac.compare_digest(name, APP_USER) and hmac.compare_digest(password, APP_PASSWORD)

    def respond(self, status, data):
        payload = json.dumps(data, allow_nan=False).encode('utf-8')
        self.send_bytes(status, payload, 'application/json; charset=utf-8')

    def send_bytes(self, status, data, mime):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def challenge(self):
        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm="ELEC 3111 Tutorial 9"')
        self.send_header('Content-Length', '0')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()

    def require_auth(self):
        if self.auth_ok():
            return True
        self.challenge()
        return False

    def route(self):
        return urlsplit(self.path).path

    def do_GET(self):
        path = self.route()
        if path == '/health':
            return self.respond(200, dict(ok=True, backend='Python cloud-ready',
                                          mode='DEMO_DATA' if DEMO_MODE else 'FIREBASE_LIVE',
                                          storageConfigured=bool(ref is not None),
                                          firebasePath=DB_PATH))
        if not self.require_auth():
            return
        files = {'/': 'index.html', '/mlr': 'index.html', '/mlp': 'index.html',
                 '/app.js': 'app.js', '/styles.css': 'styles.css',
                 '/plotly-latest.min.js': 'plotly-latest.min.js'}
        if path in files:
            name = files[path]
            file = BASE / 'web' / name
            mime = ('text/html; charset=utf-8' if name.endswith('.html') else
                    'text/css; charset=utf-8' if name.endswith('.css') else
                    'application/javascript; charset=utf-8')
            return self.send_bytes(200, file.read_bytes(), mime)
        try:
            query = parse_qs(urlsplit(self.path).query)
            limit = int(query.get('limit', [200])[0])
            if not 1 <= limit <= 1000:
                raise ValueError('limit must be 1–1000')
            if path == '/api/readings':
                rows = get_rows(limit)
                return self.respond(200, dict(ok=True, mode='DEMO_DATA' if DEMO_MODE else 'FIREBASE_LIVE',
                                              count=len(rows), rows=rows, latest=rows[-1] if rows else None))
            if path == '/api/csv':
                rows = get_rows(limit)
                s = io.StringIO()
                writer = csv.DictWriter(s, fieldnames=['time', 'temperature', 'humidity'])
                writer.writeheader()
                writer.writerows(rows)
                return self.send_bytes(200, s.getvalue().encode(), 'text/csv; charset=utf-8')
        except (ValueError, TypeError) as exc:
            return self.respond(400, dict(ok=False, error=str(exc)))
        except Exception:
            print('ERROR: Firebase read failed; check rules/indexing/credentials.', flush=True)
            return self.respond(502, dict(ok=False, error='Firebase read failed; check server logs.'))
        return self.respond(404, dict(ok=False, error='Not found'))

    def do_POST(self):
        if not self.require_auth():
            return
        path = self.route()
        if path not in ('/api/mlr', '/api/mlp', '/api/preprocess'):
            return self.respond(404, dict(ok=False, error='Not found'))
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if size < 1 or size > 65536:
                return self.respond(413, dict(ok=False, error='Invalid request size'))
            opts = json.loads(self.rfile.read(size))
            result = analyze(path.rsplit('/', 1)[-1], opts)
            return self.respond(200, dict(ok=True, result=result,
                                          mode='DEMO_DATA' if DEMO_MODE else 'FIREBASE_LIVE'))
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            return self.respond(400, dict(ok=False, error=str(exc)))
        except Exception:
            print('ERROR: Analysis failed; check input or database.', flush=True)
            return self.respond(502, dict(ok=False, error='Analysis failed. See server logs.'))


if __name__ == '__main__':
    print(f'Cloud analytics listening on 0.0.0.0:{PORT}; mode={"DEMO_DATA" if DEMO_MODE else "FIREBASE_LIVE"}; path={DB_PATH}', flush=True)
    ThreadingHTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
