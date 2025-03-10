from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from .model import PersonCreate, PersonResponse, Person
from .db import get_db


router = APIRouter()


@router.get("/person/{id}", response_model=PersonResponse)
async def get_person(person_id: int, db: Session = Depends(get_db)):
    """
    Get person details bperson id
    """
    person = db.query(Person).filter(Person.id == person_id).first()

    if person is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return {
        "data": person,
        "message": "User details fetched successfully",
        "status": status.HTTP_200_OK
    }


@router.post("/person", response_model=PersonResponse)
async def create_person(person: PersonCreate, db: Session = Depends(get_db)):
    """
    Create new person
    """
    new_person = Person(name=person.name, email=person.email,
                        is_active=person.is_active)
    db.add(new_person)
    db.commit()
    db.refresh(new_person)
    return {
        "data": new_person,
        "message": "User created successfully",
        "status": status.HTTP_201_CREATED
    }
