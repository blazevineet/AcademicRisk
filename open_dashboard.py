"""
Helper script to launch the interactive Prediction Web Panel and dashboard in your default browser.
"""

import os
import sys
import webbrowser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_PATH = os.path.join(BASE_DIR, 'results', 'report_dashboard.html')

if __name__ == '__main__':
    if not os.path.exists(DASHBOARD_PATH):
        print(f"[-] Dashboard not found at: {DASHBOARD_PATH}")
        print("[!] Generating model artifacts and dashboard first...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "run_experiments.py")])

    # Check if user requested static file mode
    if '--static' in sys.argv:
        print(f"[+] Opening static dashboard file: {DASHBOARD_PATH}")
        webbrowser.open(f"file://{os.path.abspath(DASHBOARD_PATH)}")
    else:
        # Launch the live interactive web server (serves on http://127.0.0.1:5000 with REST API)
        print("[+] Launching Academic Risk Interactive Web Server & Panel...")
        from app import run_server
        run_server(open_browser=True)
