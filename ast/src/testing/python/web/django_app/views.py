import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from db import db_session, get_person_by_id, create_new_person
from model import CreateOrEditPerson


@require_http_methods(["GET"])
def get_person(request, id):
    """Get a person by ID"""
    with db_session() as db:
        person = get_person_by_id(db, id)
        if person is None:
            return JsonResponse({'error': 'Person not found'}, status=404)
        return JsonResponse({
            'id': person.id,
            'name': person.name,
            'email': person.email
        })


@csrf_exempt
@require_http_methods(["POST"])
def create_person(request):
    """Create a new person"""
    try:
        data = json.loads(request.body)
        if not data or 'name' not in data or 'email' not in data:
            return JsonResponse({'error': 'Missing name or email'}, status=400)

        person_data = CreateOrEditPerson(
            name=data['name'], email=data['email'])

        with db_session() as db:
            new_person = create_new_person(db, person_data)
            return JsonResponse({
                'id': new_person.id,
                'name': new_person.name,
                'email': new_person.email
            }, status=201)
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
