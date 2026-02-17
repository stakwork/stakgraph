from django.urls import path
from django_app import views

urlpatterns = [
    path('person/<int:id>/', views.get_person, name='get_person'),
    path('person/', views.create_person, name='create_person'),
]
