export class Calculator {
  private result: number = 0;

  add(a: number, b: number): number {
    this.result = a + b;
    return this.result;
  }

  multiply(a: number, b: number): number {
    this.result = a * b;
    return this.result;
  }

  subtract(a: number, b: number): number {
    this.result = a - b;
    return this.result;
  }

  getResult(): number {
    return this.result;
  }
}
