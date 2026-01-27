from datetime import datetime

def log_request(method: str, path: str):
    """Simple logging middleware"""
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {method} {path}")
