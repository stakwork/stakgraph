import unittest
from mylib.api.routes import router

class TestAPI(unittest.TestCase):
    def test_create_item_route(self):
        response = router.dispatch("POST", "/items", {"id": 1, "name": "Route Item"})
        self.assertEqual(response["status"], "created")
        self.assertEqual(response["id"], 1)

    def test_404(self):
        response = router.dispatch("GET", "/invalid")
        self.assertIn("error", response)

if __name__ == '__main__':
    unittest.main()
