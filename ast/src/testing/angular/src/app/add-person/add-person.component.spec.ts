import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AddPersonComponent } from './add-person.component';
// @ast node: UnitTest "AddPersonComponent"
// @ast edge: Calls -> Class "AddPersonComponent" "add-person.component.ts"
// @ast node: Import "import-imports-srctestingangularsrcappaddpersonaddpersoncomponentspects-0"

describe('AddPersonComponent', () => {
  let component: AddPersonComponent;
  let fixture: ComponentFixture<AddPersonComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AddPersonComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AddPersonComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
