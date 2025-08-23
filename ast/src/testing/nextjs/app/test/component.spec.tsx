import { render } from '@testing-library/react';
import Card from '../../components/ui/card';

test('unit: Card export exists', () => {
  render(Card as any);
});
