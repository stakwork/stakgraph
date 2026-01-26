import hashlib
import time

def generate_id(name: str) -> str:
    """Generates a deterministic ID based on the name."""
    return hashlib.sha256(name.encode()).hexdigest()[:8]

def log_process(func):
    """Decorator to log function execution time."""
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f"Starting {func.__name__}...")
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Finished {func.__name__} in {end - start:.4f}s")
        return result
    return wrapper
