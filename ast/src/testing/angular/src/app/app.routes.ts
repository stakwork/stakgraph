import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { PeopleListComponent } from './people-list/people-list.component';
import { AddPersonComponent } from './add-person/add-person.component';
// @ast node: Var "routes"
// @ast node: Class "AppRoutingModule"
// @ast node: Import "import-imports-srctestingangularsrcappapproutests-0"

export const routes: Routes = [
  { path: '', redirectTo: '/add-person', pathMatch: 'full' },
  { path: 'people', component: PeopleListComponent },
  { path: 'add-person', component: AddPersonComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
