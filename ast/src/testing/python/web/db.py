from contextlib import contextmanager
from sqlalchemy.exc import IntegrityError
from database import SessionLocal, engine
from model import Person, CreateOrEditPerson


def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        print(f"Error accessing database: {e}")
        raise
    finally:
        db.close()


@contextmanager
def db_session():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Database error: {e}")
        raise
    finally:
        db.close()


def get_person_by_id(db, person_id: int):
    """Get a person by their ID"""
    return db.query(Person).filter(Person.id == person_id).first()


def create_new_person(db, person_data: CreateOrEditPerson):
    """Create a new person in the database"""
    try:
        new_person = Person(
            name=person_data.name,
            email=person_data.email
        )
        db.add(new_person)
        db.commit()
        db.refresh(new_person)
        return new_person
    except IntegrityError as e:
        db.rollback()
        if "UNIQUE constraint failed: person.email" in str(e):
            raise ValueError(
                f"A person with email {person_data.email} already exists")
        elif "UNIQUE constraint failed: person.id" in str(e):
            raise ValueError(f"ID conflict - please try again")
        else:
            raise
