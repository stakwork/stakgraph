import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { PeopleListComponent } from './people-list/people-list.component';
import { AddPersonComponent } from './add-person/add-person.component';
import { PersonItemComponent } from './person-item/person-item.component';

export const routes: Routes = [
  { path: '', redirectTo: '/add-person', pathMatch: 'full' },
  { path: 'people', component: PeopleListComponent },
  { path: 'add-person', component: AddPersonComponent },
  { path: 'person/:id', component: PersonItemComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
