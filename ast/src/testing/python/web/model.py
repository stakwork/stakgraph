from sqlalchemy import Column, Integer, String
from database import Base
from pydantic import BaseModel
from typing import Optional
from abc import ABC, abstractmethod
from functools import lru_cache


class Person(Base):
    """
    Person model for storing user details
    """
    __tablename__ = "person"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, index=True)

    def __repr__(self):
        return f"<Person {self.name}>"

    def __str__(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
        }


class CreateOrEditPerson(BaseModel):
    """
    PersonCreate model for creating new person
    """
    id: Optional[int] = None
    name: str
    email: str


class PersonResponse(BaseModel):
    """
    PersonResponse model for returning person details
    """
    id: int
    name: str
    email: str


class Animal(ABC):
    """Abstract base class for animals"""

    @abstractmethod
    def make_sound(self) -> str:
        pass

    @abstractmethod
    def move(self) -> str:
        pass


class Dog(Animal):
    """Concrete class implementing the Animal ABC."""

    def make_sound(self) -> str:
        return "Woof!"

    def move(self) -> str:
        return "Runs on four legs"

    @property
    def species(self) -> str:
        return "Canis familiaris"

    @staticmethod
    def is_mammal() -> bool:
        return True

    @classmethod
    def create_puppy(cls) -> "Dog":
        return cls()


@lru_cache(maxsize=128)
def get_animal_info(animal_type: str) -> str:
    """Cached function to get animal information"""
    if animal_type == "dog":
        return "Dogs are loyal companions"
    elif animal_type == "cat":
        return "Cats are independent"
    return "Unknown animal"
