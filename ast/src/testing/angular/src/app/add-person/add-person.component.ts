import { Component } from '@angular/core';
import { Router } from '@angular/router'; // Import the Router
import { PeopleService } from '../people.service';
import { Person } from '../models/person.model';
import { FormsModule } from '@angular/forms';


@Component({
  selector: 'app-add-person',
  templateUrl: './add-person.component.html',
  styleUrls: ['./add-person.component.css'],
  imports: [FormsModule]
})
export class AddPersonComponent {
  name: string = '';
  age: number | null = null;

  constructor(private peopleService: PeopleService, private router: Router) {}

  addPerson() {
    if (this.name && this.age != null) {
      const newPerson: Person = {
        id: 0,
        name: this.name,
        age: this.age
      };
      this.peopleService.addPerson(newPerson);
      this.name = '';
      this.age = null;


      this.router.navigate(['/']);
    }
  }
  
  navigateToHome() {
    this.router.navigate(['/']);
  }
}
