
import unittest
from model import Person, create_puppy, get_animal_info

class TestPerson(unittest.TestCase):
    def test_person_creation(self):
        p = Person(name="Alice", age=30)
        self.assertEqual(p.name, "Alice")
        self.assertEqual(p.age, 30)

    def test_person_str(self):
        p = Person(name="Bob", age=25)
        self.assertEqual(str(p), "Person(name=Bob, age=25)")

def test_puppy_creation():
    # Pytest style
    puppy = create_puppy("Buddy")
    assert puppy == "Puppy named Buddy"

def test_animal_info_cache():
    # Testing cached function
    info1 = get_animal_info("dog")
    info2 = get_animal_info("dog")
    assert info1 == info2
