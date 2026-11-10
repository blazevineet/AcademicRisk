"""
Helper script to open the interactive HTML report dashboard in your default web browser.
"""

import os
import webbrowser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_PATH = os.path.join(BASE_DIR, 'results', 'report_dashboard.html')

if __name__ == '__main__':
    if not os.path.exists(DASHBOARD_PATH):
        print(f"[-] Dashboard not found at: {DASHBOARD_PATH}")
        print("[!] Generating report dashboard first...")
        import subprocess
        subprocess.run(["python", os.path.join(BASE_DIR, "run_experiments.py")])
        
    print(f"[+] Opening dashboard: {DASHBOARD_PATH}")
    webbrowser.open(f"file://{os.path.abspath(DASHBOARD_PATH)}")
