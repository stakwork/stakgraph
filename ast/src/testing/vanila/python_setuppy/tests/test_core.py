import unittest
from mylib.core import Processor

class TestProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = Processor()

    def test_process_valid_data(self):
        data = {
            "items": [
                {"id": 1, "name": "Test Item", "tags": ["test"]}
            ]
        }
        result = self.processor.process(data)
        self.assertTrue(result.success)
        self.assertEqual(result.processed_count, 1)

    def test_process_empty_data(self):
        result = self.processor.process({})
        self.assertFalse(result.success)

if __name__ == '__main__':
    unittest.main()
