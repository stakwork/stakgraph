import { Component, Input, Output, EventEmitter, OnInit } from '@angular/core';
import { Person } from '../models/person.model';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { PeopleService } from '../people.service';

@Component({
  selector: 'app-person-item',
  templateUrl: './person-item.component.html',
  styleUrls: ['./person-item.component.css'],
  standalone: true,
  imports: [CommonModule, RouterModule]
})
export class PersonItemComponent implements OnInit {
  @Input() person!: Person;
  @Output() delete = new EventEmitter<number>();
  isStandalone = false;

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private peopleService: PeopleService
  ) {}

  ngOnInit() {
    const id = this.route.snapshot.paramMap.get('id');
    if (id) {
      this.isStandalone = true;
      this.peopleService.people$.subscribe(people => {
        const personFound = people.find(p => p.id === +id);
        if (personFound) {
          this.person = personFound;
        }
      });
    }
  }

  onDelete() {
    this.delete.emit(this.person.id);
    if (this.isStandalone) {
      this.peopleService.deletePerson(this.person.id);
      this.router.navigate(['/people']);
    }
  }

  goBack() {
    this.router.navigate(['/people']);
  }
} 