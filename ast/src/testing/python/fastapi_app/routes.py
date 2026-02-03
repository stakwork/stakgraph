from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from model import CreateOrEditPerson, PersonResponse
from db import get_db, get_person_by_id, create_new_person


router = APIRouter()


@router.get("/person/{id}", response_model=PersonResponse)
async def get_person(id: int, db: Session = Depends(get_db)):
    """
    Get user details by user id
    """
    person = get_person_by_id(db, id)
    if person is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Person not found")

    # Return a dict that directly matches the PersonResponse model
    return PersonResponse(
        id=person.id,
        name=person.name,
        email=person.email
    )


# Create new user endpoint
@router.post("/person/", response_model=PersonResponse)
async def create_person(person: CreateOrEditPerson, db: Session = Depends(get_db)):
    """
    Create new user
    """
    new_person = create_new_person(db, person)

    # Return a dict that directly matches the PersonResponse model
    return PersonResponse(
        id=new_person.id,
        name=new_person.name,
        email=new_person.email
    )
