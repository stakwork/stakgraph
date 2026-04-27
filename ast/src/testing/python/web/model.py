# @ast node: Class "Person"
# @ast edge: Operand -> Function "__repr__" "model.py"
# @ast edge: Operand -> Function "__str__" "model.py"
# @ast node: Function "__repr__"
# @ast node: Function "__str__"
# @ast node: Var "__tablename__"
# @ast node: Var "id"
# @ast node: Var "name"
# @ast node: Var "email"
# @ast node: DataModel "CreateOrEditPerson"
# @ast node: Var "id"
# @ast node: Var "name"
# @ast node: Var "email"
# @ast node: DataModel "PersonResponse"
# @ast node: Var "id"
# @ast node: Var "name"
# @ast node: Var "email"
# @ast node: Trait "Animal"
# @ast node: Class "Animal"
# @ast edge: Operand -> Function "make_sound" "model.py"
# @ast edge: Operand -> Function "move" "model.py"
# @ast edge: ParentOf -> Class "Dog" "model.py"
# @ast node: Function "make_sound"
# @ast node: Function "move"
# @ast node: Class "Dog"
# @ast edge: Implements -> Trait "Animal" "model.py"
# @ast edge: Operand -> Function "make_sound" "model.py"
# @ast edge: Operand -> Function "move" "model.py"
# @ast edge: Operand -> Function "species" "model.py"
# @ast edge: Operand -> Function "is_mammal" "model.py"
# @ast edge: Operand -> Function "create_puppy" "model.py"
# @ast node: Function "make_sound"
# @ast node: Function "move"
# @ast node: Function "species"
# @ast node: Function "is_mammal"
# @ast node: Function "create_puppy"
# @ast node: Function "get_animal_info"
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
