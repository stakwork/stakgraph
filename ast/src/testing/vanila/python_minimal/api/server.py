import http.server
import socketserver
from .routes import handle_request
from .middleware import log_request

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        log_request(self.command, self.path)
        handle_request(self, "GET")

    def do_POST(self):
        log_request(self.command, self.path)
        handle_request(self, "POST")

class Server:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.httpd = None

    def run(self):
        with socketserver.TCPServer((self.host, self.port), RequestHandler) as httpd:
            self.httpd = httpd
            print(f"Serving at {self.host}:{self.port}")
            httpd.serve_forever()
