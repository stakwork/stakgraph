import { TestBed } from '@angular/core/testing';

import { PeopleService } from './people.service';
// @ast node: UnitTest "PeopleService"
// @ast edge: Calls -> Class "PeopleService" "people.service.ts"
// @ast node: Import "import-imports-srctestingangularsrcapppeopleservicespects-0"

describe('PeopleService', () => {
  let service: PeopleService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(PeopleService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
