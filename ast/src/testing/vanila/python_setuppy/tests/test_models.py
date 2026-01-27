import unittest
from mylib.models import Item
from pydantic import ValidationError

class TestModels(unittest.TestCase):
    def test_item_creation(self):
        item = Item(id=1, name="Test")
        self.assertEqual(item.name, "Test")
        self.assertEqual(item.tags, [])

    def test_item_validation(self):
        with self.assertRaises(ValidationError):
            Item(id="not-an-int", name="Test")

if __name__ == '__main__':
    unittest.main()
