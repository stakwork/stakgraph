import { Component, OnInit } from '@angular/core';
import { PeopleService } from '../people.service';
import { Person } from '../models/person.model';
import { CommonModule } from '@angular/common';
import { PersonItemComponent } from '../person-item/person-item.component';
import { RouterModule } from '@angular/router';


@Component({
  selector: 'app-people-list',
  templateUrl: './people-list.component.html',
  styleUrls: ['./people-list.component.css'],
  standalone: true,
  imports: [ CommonModule, PersonItemComponent, RouterModule ]
})
export class PeopleListComponent implements OnInit {
  people: Person[] = [];

  constructor(private peopleService: PeopleService) {}

  ngOnInit() {
    this.peopleService.people$.subscribe(people => {
      this.people = people;
    });
  }

  deletePerson(id: number) {
    this.peopleService.deletePerson(id);
  }
}
