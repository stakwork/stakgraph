import { ComponentFixture, TestBed } from '@angular/core/testing';
import { PersonItemComponent } from './person-item.component';
import { Person } from '../models/person.model';

describe('PersonItemComponent', () => {
  let component: PersonItemComponent;
  let fixture: ComponentFixture<PersonItemComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PersonItemComponent]
    }).compileComponents();

    fixture = TestBed.createComponent(PersonItemComponent);
    component = fixture.componentInstance;
    component.person = { id: 1, name: 'John Doe', age: 30 };
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should display person name and age', () => {
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.textContent).toContain('John Doe');
    expect(compiled.textContent).toContain('30');
  });

  it('should emit delete event when delete button is clicked', () => {
    spyOn(component.delete, 'emit');
    const button = fixture.nativeElement.querySelector('.delete-btn');
    button.click();
    expect(component.delete.emit).toHaveBeenCalledWith(1);
  });
}); 