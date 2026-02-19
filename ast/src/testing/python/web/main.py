import os
import signal
import subprocess
import sys
from fastapi import FastAPI
from flask import Flask
from fastapi_app.routes import router
from database import Base, engine
from flask_app.routes import flask_bp


processes = []


address = "0.0.0.0"
ports = {
    "fastapi": 8000,
    "flask": 5000,
    "django": 7000
}


Base.metadata.create_all(bind=engine)


fastapi_app = FastAPI(
    title="Python Test Example StakGraph",
    description="A simple Treesitter test example for Python"
)
fastapi_app.include_router(router)


flask_app = Flask(__name__)
flask_app.register_blueprint(flask_bp)


def cleanup():
    """Clean up all processes"""
    for process in processes:
        if process.poll() is None:  # Process is still running
            process.terminate()


def signal_handler(sig, frame):
    print("\nReceived termination signal. Shutting down...")
    cleanup()
    cleanup()
    sys.exit(0)


# This function runs the servers
def run_servers():
    """Run all three frameworks"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = current_dir + ":" + env.get("PYTHONPATH", "")

    fastapi_cmd = ["uvicorn", "main:fastapi_app",
                   "--host", address, "--port", str(ports["fastapi"])]

    flask_cmd = ["python", "-c",
                 f"from main import flask_app; flask_app.run(host='{address}', port={ports['flask']})"]

    django_cmd = ["python", "manage.py",
                  "runserver", f"{address}:{ports['django']}", "--noreload"]

    try:

        fastapi_process = subprocess.Popen(fastapi_cmd, env=env)
        processes.append(fastapi_process)

        flask_process = subprocess.Popen(flask_cmd, env=env)
        processes.append(flask_process)

        django_process = subprocess.Popen(django_cmd, env=env)
        processes.append(django_process)

        print(f"FastAPI server running on http://localhost:{ports['fastapi']}")
        print(f"Flask server running on http://localhost:{ports['flask']}")
        print(f"Django server running on http://localhost:{ports['django']}")

        fastapi_process.wait()
        flask_process.wait()
        django_process.wait()

    except KeyboardInterrupt:
        print("\nShutting down servers...")
        cleanup()
        sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    run_servers()
