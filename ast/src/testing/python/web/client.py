# @ast node: Function "fetch_person"
# @ast node: Request "http://localhost:5000/person/1" [verb=GET]
# @ast node: Function "create_person"
# @ast node: Request "http://localhost:5000/person/" [verb=POST]
# @ast edge: Calls -> Endpoint "/person/" "fastapi_app/routes.py" [verb=POST]
# @ast edge: Calls -> Endpoint "person/" "django_app/urls.py" [verb=POST]
# @ast node: Function "delete_person"
# @ast node: Request "http://localhost:5000/person/1" [verb=DELETE]
import requests
import httpx


def fetch_person():
    return requests.get("http://localhost:5000/person/1")


def create_person(payload):
    return requests.post("http://localhost:5000/person/", json=payload)


def delete_person():
    return httpx.delete("http://localhost:5000/person/1")
