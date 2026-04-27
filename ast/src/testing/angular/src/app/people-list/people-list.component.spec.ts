import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PeopleListComponent } from './people-list.component';
// @ast node: UnitTest "PeopleListComponent"
// @ast edge: Calls -> Class "PeopleListComponent" "people-list.component.ts"
// @ast node: Import "import-imports-srctestingangularsrcapppeoplelistpeoplelistcomponentspects-0"

describe('PeopleListComponent', () => {
  let component: PeopleListComponent;
  let fixture: ComponentFixture<PeopleListComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PeopleListComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PeopleListComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
