import { Calculator } from '../../lib/calculator';

describe('unit: Calculator class', () => {
  it('performs arithmetic operations', () => {
    const calc = new Calculator();
    const sum = calc.add(10, 5);
    expect(sum).toBe(15);
    
    const product = calc.multiply(4, 3);
    expect(product).toBe(12);
    
    const difference = calc.subtract(20, 8);
    expect(difference).toBe(12);
    
    const result = calc.getResult();
    expect(result).toBe(12);
  });
});
