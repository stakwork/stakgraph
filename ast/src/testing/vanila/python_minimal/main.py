import sys
from api.server import Server
from config import settings
from database import db

def main():
    print(f"Starting application on port {settings.PORT}...")
    
    if not db.connect():
        print("Failed to connect to database.")
        sys.exit(1)

    server = Server(host="0.0.0.0", port=settings.PORT)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()
