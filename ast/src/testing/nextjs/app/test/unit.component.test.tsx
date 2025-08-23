import { render, screen } from '@testing-library/react';
import { Button } from '../../components/ui/button';

describe('unit: Button component', () => {
  it('renders variant', () => {
    render(<Button variant="default">Click</Button>);
    screen.getByText('Click');
  });
});
