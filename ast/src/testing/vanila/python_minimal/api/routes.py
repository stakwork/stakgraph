import json
from http.server import BaseHTTPRequestHandler
from database import db
from utils.helpers import generate_token

def handle_request(handler: BaseHTTPRequestHandler, method: str):
    path = handler.path
    
    if path == "/" and method == "GET":
        return send_json(handler, 200, {"message": "Welcome to Python Minimal API"})
    
    if path == "/health" and method == "GET":
        return send_json(handler, 200, {"status": "ok"})
        
    if path == "/users" and method == "GET":
        users = db.all("users")
        return send_json(handler, 200, {"users": users})
    
    if path == "/token" and method == "POST":
        return send_json(handler, 201, {"token": generate_token()})

    return send_json(handler, 404, {"error": "Not Found"})

def send_json(handler: BaseHTTPRequestHandler, status: int, data: dict):
    handler.send_response(status)
    handler.send_header("Content-type", "application/json")
    handler.end_headers()
    handler.wfile.write(json.dumps(data).encode("utf-8"))
