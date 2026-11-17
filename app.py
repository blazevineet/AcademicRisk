"""
Web Application & Interactive Prediction Server
Academic Risk & Engagement Prediction System (Units 1-5)

Provides:
- Interactive Web Interface for real-time risk prediction
- REST API endpoint POST /api/predict backed by Champion Bayesian Logistic Regression & HMM Viterbi
- REST API endpoint GET /api/sample/<preset> for quick sample student data
- Static asset serving for benchmark graphs and evaluation dashboards
- Multi-threaded HTTP server with auto-port selection and browser launch
"""

import os
import sys
import json
import socket
import threading
import webbrowser
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DASHBOARD_FILE = os.path.join(RESULTS_DIR, 'report_dashboard.html')

# Import the core predictor
from predict import predict_student, sanitize_student_input

SAMPLE_PROFILES = {
    'high_risk': {
        'name': 'Student A (High Hazard Profile)',
        'data': {
            'Age': 21,
            'Gender': 'Male',
            'Socio_Economic_Status': 'Low',
            'GPA': 5.38,
            'Attendance_Rate': 58.0,
            'Assignments_Submitted': 4,
            'Online_Learning_Hours': 5.2,
            'Forum_Participation': 2,
            'Lab_Performance': 52.0,
            'Extracurricular_Participation': 'No'
        }
    },
    'safe': {
        'name': 'Student B (High Performing / Safe Profile)',
        'data': {
            'Age': 20,
            'Gender': 'Female',
            'Socio_Economic_Status': 'High',
            'GPA': 9.62,
            'Attendance_Rate': 94.0,
            'Assignments_Submitted': 10,
            'Online_Learning_Hours': 22.5,
            'Forum_Participation': 24,
            'Lab_Performance': 91.0,
            'Extracurricular_Participation': 'Yes'
        }
    },
    'borderline': {
        'name': 'Student C (Borderline Vulnerable Profile)',
        'data': {
            'Age': 22,
            'Gender': 'Male',
            'Socio_Economic_Status': 'Medium',
            'GPA': 6.75,
            'Attendance_Rate': 72.5,
            'Assignments_Submitted': 6,
            'Online_Learning_Hours': 9.0,
            'Forum_Participation': 5,
            'Lab_Performance': 68.0,
            'Extracurricular_Participation': 'No'
        }
    }
}

class RiskMLRequestHandler(SimpleHTTPRequestHandler):
    """Custom request handler with REST API and static asset routing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def end_headers(self):
        # Enable CORS for local integration
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def _send_json(self, payload: dict, status=HTTPStatus.OK):
        body = json.dumps(payload, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Root route: serve main interactive dashboard
        if path in ('/', '/index.html', '/dashboard', '/predict'):
            if os.path.exists(DASHBOARD_FILE):
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                with open(DASHBOARD_FILE, 'rb') as f:
                    content = f.read()
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Dashboard HTML not found. Please run run_experiments.py first.")
                return

        # Health endpoint
        if path == '/api/health':
            self._send_json({
                'status': 'healthy',
                'service': 'Academic Risk Prediction API',
                'champion_model': 'Bayesian Logistic Regression (Unit 2 Champion)'
            })
            return

        # Sample profiles endpoint
        if path.startswith('/api/sample'):
            parts = path.strip('/').split('/')
            key = parts[-1] if len(parts) >= 3 else None
            if key and key in SAMPLE_PROFILES:
                self._send_json({'success': True, 'profile': SAMPLE_PROFILES[key]})
            else:
                self._send_json({'success': True, 'samples': SAMPLE_PROFILES})
            return

        # Direct file request in results folder fallback
        if path.startswith('/results/'):
            filename = os.path.basename(path)
            file_on_disk = os.path.join(RESULTS_DIR, filename)
            if os.path.exists(file_on_disk) and os.path.isfile(file_on_disk):
                ext = os.path.splitext(filename)[1].lower()
                content_types = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.html': 'text/html; charset=utf-8',
                    '.json': 'application/json; charset=utf-8',
                    '.csv': 'text/csv; charset=utf-8',
                    '.js': 'application/javascript; charset=utf-8',
                    '.css': 'text/css; charset=utf-8',
                }
                ctype = content_types.get(ext, 'application/octet-stream')
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', ctype)
                with open(file_on_disk, 'rb') as f:
                    data = f.read()
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

        # Also support images referenced by root filename e.g. /viterbi_trajectories.png
        possible_in_results = os.path.join(RESULTS_DIR, path.lstrip('/'))
        if os.path.exists(possible_in_results) and os.path.isfile(possible_in_results):
            ext = os.path.splitext(possible_in_results)[1].lower()
            if ext == '.png':
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'image/png')
                with open(possible_in_results, 'rb') as f:
                    data = f.read()
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

        # Standard file serving fallback
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == '/api/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_json({'success': False, 'error': 'Empty request body'}, status=HTTPStatus.BAD_REQUEST)
                return

            raw_body = self.rfile.read(content_length).decode('utf-8')
            try:
                # Support both JSON payload and Form-encoded payload
                if self.headers.get('Content-Type', '').startswith('application/x-www-form-urlencoded'):
                    parsed_form = urllib.parse.parse_qs(raw_body)
                    data = {k: v[0] for k, v in parsed_form.items()}
                else:
                    data = json.loads(raw_body)
            except Exception as e:
                self._send_json({'success': False, 'error': f'Invalid request data: {str(e)}'}, status=HTTPStatus.BAD_REQUEST)
                return

            try:
                result = predict_student(data)
                self._send_json({
                    'success': True,
                    'input': sanitize_student_input(data),
                    'result': result
                })
            except Exception as e:
                self._send_json({'success': False, 'error': f'Prediction execution error: {str(e)}'}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self.send_error(HTTPStatus.NOT_FOUND, f"Endpoint {path} not found.")

def find_available_port(start_port=5000, max_attempts=20):
    """Finds an open port starting from start_port."""
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', p))
                return p
            except OSError:
                continue
    return start_port

def run_server(port=None, open_browser=True):
    """Starts the multi-threaded web server."""
    if port is None:
        port = find_available_port(5000)

    server_address = ('127.0.0.1', port)
    httpd = ThreadingHTTPServer(server_address, RiskMLRequestHandler)

    url = f"http://127.0.0.1:{port}"
    print("=" * 80)
    print(" ACADEMIC RISK PREDICTION SYSTEM - INTERACTIVE WEB SERVER")
    print("=" * 80)
    print(f" [+] Web App & Interactive Panel URL:  {url}")
    print(f" [+] REST Prediction API:               {url}/api/predict")
    print(f" [+] API Health Check:                  {url}/api/health")
    print(f" [+] Sample Profiles:                   {url}/api/sample")
    print("=" * 80)
    print(" Press Ctrl+C in terminal to stop server.\n")

    if open_browser:
        def launch():
            import time
            time.sleep(0.6)
            webbrowser.open(url)
        threading.Thread(target=launch, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Shutting down server gracefully...")
        httpd.server_close()
        sys.exit(0)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Academic Risk ML Web Server & Interactive Panel")
    parser.add_argument('--port', type=int, default=None, help="Port to run server on")
    parser.add_argument('--no-browser', action='store_true', help="Do not open browser automatically")
    args = parser.parse_args()

    run_server(port=args.port, open_browser=not args.no_browser)
