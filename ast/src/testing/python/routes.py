from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Person(BaseModel):
    name: str
    age: int

# Endpoint to get a person by GET request
@app.get("/person", response_model=Person)
def get_person():
    person = Person(name="John Doe", age=30)
    return person

# Endpoint to create a new person by POST request
@app.post("/person", response_model=Person)
def create_person(person: Person):
    return person
