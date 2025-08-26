import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DataAnalysis } from './data-analysis';

describe('DataAnalysis', () => {
  let component: DataAnalysis;
  let fixture: ComponentFixture<DataAnalysis>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DataAnalysis]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DataAnalysis);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
