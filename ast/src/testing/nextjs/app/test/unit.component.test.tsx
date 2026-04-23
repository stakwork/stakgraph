import { render, screen } from '@testing-library/react';
import { Button } from '../../components/ui/button';

// @ast node: UnitTest "unit: Button component"
// @ast edge: Contains <- File "unit.component.test.tsx" "src/testing/nextjs/app/test/unit.component.test.tsx"
// @ast edge: Calls -> Function "Button" "src/testing/nextjs/components/ui/button.tsx"
describe('unit: Button component', () => {
  it('renders variant', () => {
    render(<Button variant="default">Click</Button>);
    screen.getByText('Click');
  });
});
