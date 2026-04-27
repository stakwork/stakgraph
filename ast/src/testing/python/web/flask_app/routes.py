# @ast node: Endpoint "/person/<int:id>"
# @ast edge: Handler -> Function "get_person" "flask_app/routes.py"
# @ast node: Endpoint "/person/"
# @ast edge: Handler -> Function "create_person" "flask_app/routes.py"
# @ast node: Var "flask_bp"
# @ast node: Function "get_person"
# @ast edge: Calls -> Function "db_session" "db.py"
# @ast edge: Calls -> Function "get_person_by_id" "db.py"
# @ast node: Function "create_person"
# @ast edge: Calls -> Function "db_session" "db.py"
# @ast edge: Calls -> Function "create_new_person" "db.py"
from flask import Blueprint, request, jsonify
from db import db_session, get_person_by_id, create_new_person
from model import CreateOrEditPerson

# Create a Blueprint for Flask routes
flask_bp = Blueprint('flask_routes', __name__)


@flask_bp.route('/person/<int:id>', methods=['GET'])
def get_person(id):
    with db_session() as db:
        person = get_person_by_id(db, id)
        if person is None:
            return jsonify({'error': 'Person not found'}), 404
        return jsonify({'person': {'id': person.id, 'name': person.name, 'email': person.email}}), 200


@flask_bp.route('/person/', methods=['POST'])
def create_person():
    # Get data from request
    data = request.get_json()
    if not data or not 'name' in data or not 'email' in data:
        return jsonify({'error': 'Missing name or email'}), 400

    # Create person data object
    person_data = CreateOrEditPerson(name=data['name'], email=data['email'])

    with db_session() as db:
        # Create new person
        new_person = create_new_person(db, person_data)
        return jsonify({'id': new_person.id, 'name': new_person.name, 'email': new_person.email}), 201
