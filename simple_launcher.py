"""
Simple Professional AI Suite Launcher
Clean deployment without Unicode issues
"""

import subprocess
import time
import sys
import os

def start_professional_app():
    """Start the professional AI app"""
    print("Starting Professional AI Suite on port 8506...")
    
    cmd = [
        'streamlit', 'run', 'professional_ai_app.py',
        '--server.port', '8506',
        '--server.headless', 'true',
        '--server.fileWatcherType', 'none',
        '--browser.gatherUsageStats', 'false'
    ]
    
    process = subprocess.Popen(cmd, cwd=os.getcwd())
    return process

def start_enhanced_app():
    """Start the enhanced ML app"""
    print("Starting Enhanced ML Pipeline on port 8504...")
    
    cmd = [
        'streamlit', 'run', 'enhanced_app.py',
        '--server.port', '8504',
        '--server.headless', 'true',
        '--server.fileWatcherType', 'none',
        '--browser.gatherUsageStats', 'false'
    ]
    
    process = subprocess.Popen(cmd, cwd=os.getcwd())
    return process

def main():
    print("=" * 60)
    print("PROFESSIONAL AI SUITE LAUNCHER")
    print("=" * 60)
    
    # Start both applications
    prof_process = start_professional_app()
    time.sleep(3)
    
    enhanced_process = start_enhanced_app()
    time.sleep(5)
    
    print("\nApplications started successfully!")
    print("\nACCESS URLS:")
    print("Professional AI Suite: http://localhost:8506")
    print("Enhanced ML Pipeline:  http://localhost:8504")
    print("\nPress Ctrl+C to stop all applications")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping applications...")
        prof_process.terminate()
        enhanced_process.terminate()
        print("Applications stopped.")

if __name__ == "__main__":
    main()
