from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Database class definition
class Database:
    def __init__(self, name):
        self.name = name

    def connect(self):
        print(f"Connecting to {self.name} database...")

    def fetch_data(self):
        print("Fetching data from database...")

# Person class for in-memory data storage
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def get_details(self):
        return {"name": self.name, "age": self.age}

# SQLAlchemy-based Person class definition for database
class PersonDB(Base):
    __tablename__ = 'persons'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return f"PersonDB(name={self.name}, age={self.age})"
