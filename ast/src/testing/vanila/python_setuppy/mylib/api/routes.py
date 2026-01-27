from .handlers import ItemHandler

class Router:
    def __init__(self):
        self.handler = ItemHandler()
        self.routes = {
            "POST /items": self.handler.create_item,
            "GET /items/{id}": self.handler.get_item
        }

    def dispatch(self, method: str, path: str, body: dict = None):
        key = f"{method} {path}"
        if key in self.routes:
            return self.routes[key](body) if body else self.routes[key](1) # Simple mock dispatch
        return {"error": "Route not found"}

router = Router()
