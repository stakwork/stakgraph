

import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { Person } from './models/person.model';

function LogMethod(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
  const originalMethod = descriptor.value;
  descriptor.value = function (...args: any[]) {
    console.log(`Calling ${propertyKey} with`, args);
    return originalMethod.apply(this, args);
  };
  return descriptor;
}

@Injectable({
  providedIn: 'root',
})
export class PeopleService {
  private peopleSubject = new BehaviorSubject<Person[]>([]);
  people$ = this.peopleSubject.asObservable();

  private people: Person[] = [];

  constructor() {}

  @LogMethod
  addPerson(person: Person): void {
    this.people.push(person);
    this.peopleSubject.next(this.people);
  }

  @LogMethod
  deletePerson(id: number): void {
    this.people = this.people.filter(person => person.id !== id);
    this.peopleSubject.next(this.people);
  }

  getPeople(): Person[] {
    return this.people;
  }

}
