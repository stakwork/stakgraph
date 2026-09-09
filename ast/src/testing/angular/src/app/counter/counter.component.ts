import { Component, signal, computed } from '@angular/core';
// @ast node: Class "CounterComponent"
// @ast node: Var "count"
// @ast node: Var "doubled"
// @ast node: Function "increment"
// @ast node: Import "import-imports-srctestingangularsrcappcountercountercomponentts-0"

@Component({
  selector: 'app-counter',
  template: '<p>{{ count() }}</p>'
})
export class CounterComponent {
  count = signal(0);
  doubled = computed(() => this.count() * 2);

  increment() {
    this.count.set(this.count() + 1);
  }
}
