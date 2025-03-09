from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from models import Base, Person
from utils import Calculator, greet

DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/greet/{name}")
async def greet_endpoint(name: str):
    return {"message": greet(name)}

@app.post("/calculate")
async def calculate_endpoint(a: int, b: int):
    calculator = Calculator()
    result = calculator.add(a, b)
    return {"result": result}

@app.post("/person/")
async def create_person(name: str, age: int, db: Session = Depends(get_db)):
    person = Person(name=name, age=age)
    db.add(person)
    db.commit()
    db.refresh(person)
    return person

@app.get("/person/{person_id}")
async def read_person(person_id: int, db: Session = Depends(get_db)):
    person = db.query(Person).filter(Person.id == person_id).first()
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return person
