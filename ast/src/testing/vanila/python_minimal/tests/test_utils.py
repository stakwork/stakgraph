import unittest
from utils.helpers import slugify
from utils.validators import is_valid_email

class TestUtils(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(slugify("Hello World"), "hello-world")
    
    def test_email_validator(self):
        self.assertTrue(is_valid_email("test@example.com"))
        self.assertFalse(is_valid_email("invalid-email"))

if __name__ == '__main__':
    unittest.main()
