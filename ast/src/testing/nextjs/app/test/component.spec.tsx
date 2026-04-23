import { render } from '@testing-library/react';
import Card from '../../components/ui/card';

// @ast node: UnitTest "unit: Card export exists"
// @ast edge: Contains <- File "component.spec.tsx" "src/testing/nextjs/app/test/component.spec.tsx"
test('unit: Card export exists', () => {
  render(Card as any);
});
