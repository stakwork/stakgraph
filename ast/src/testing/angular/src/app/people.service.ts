

import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { Person } from './models/person.model';
// @ast node: Class "PeopleService"
// @ast node: Function "LogMethod"
// @ast edge: Calls -> Request "console.log" "people.service.ts"
// @ast edge: Calls -> Request "originalMethod.apply" "people.service.ts"
// @ast node: Function "constructor"
// @ast node: Function "addPerson"
// @ast edge: Calls -> Request "this.people.push" "people.service.ts"
// @ast edge: Calls -> Request "this.peopleSubject.next" "people.service.ts"
// @ast node: Function "deletePerson"
// @ast edge: Calls -> Request "this.people.filter" "people.service.ts"
// @ast edge: Calls -> Request "this.peopleSubject.next" "people.service.ts"
// @ast node: Function "getPeople"
// @ast node: Import "import-imports-srctestingangularsrcapppeopleservicets-2"

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
