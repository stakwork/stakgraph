import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';
import { PeopleService } from '../people.service';
import { Person } from '../models/person.model';
import { CommonModule } from '@angular/common';
// @ast node: Class "PeopleListComponent"
// @ast node: Function "constructor"
// @ast node: Function "ngOnInit"
// @ast edge: Calls -> Request "this.peopleService.people$.subscribe" "people-list.component.ts"
// @ast node: Function "deletePerson"
// @ast edge: Calls -> Request "this.peopleService.deletePerson" "people-list.component.ts"
// @ast edge: Calls -> Request "this.personDeleted.emit" "people-list.component.ts"
// @ast node: Import "import-imports-srctestingangularsrcapppeoplelistpeoplelistcomponentts-0"
// @ast node: Page "PeopleListComponent"



@Component({
  selector: 'app-people-list',
  templateUrl: './people-list.component.html',
  styleUrls: ['./people-list.component.css'],
  imports: [ CommonModule ]

})
export class PeopleListComponent implements OnInit {
  people: Person[] = [];

  @Input()
  maxItems: number = 10;

  @Output()
  personDeleted = new EventEmitter<number>();

  constructor(private peopleService: PeopleService) {}

  ngOnInit() {
    this.peopleService.people$.subscribe(people => {
      this.people = people;
    });
  }

  deletePerson(id: number) {
    this.peopleService.deletePerson(id);
    this.personDeleted.emit(id);
  }
}
