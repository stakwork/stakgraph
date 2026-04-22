// @ast node: Class "Calculator"
// @ast edge: Contains <- File "calculator.ts" "src/testing/nextjs/lib/calculator.ts"
export class Calculator {
  private result: number = 0;

  // @ast node: Function "add"
  // @ast edge: Contains <- File "calculator.ts" "src/testing/nextjs/lib/calculator.ts"
  add(a: number, b: number): number {
    this.result = a + b;
    return this.result;
  }

  // @ast node: Function "multiply"
  // @ast edge: Contains <- File "calculator.ts" "src/testing/nextjs/lib/calculator.ts"
  multiply(a: number, b: number): number {
    this.result = a * b;
    return this.result;
  }

  // @ast node: Function "subtract"
  // @ast edge: Contains <- File "calculator.ts" "src/testing/nextjs/lib/calculator.ts"
  subtract(a: number, b: number): number {
    this.result = a - b;
    return this.result;
  }

  // @ast node: Function "getResult"
  // @ast edge: Contains <- File "calculator.ts" "src/testing/nextjs/lib/calculator.ts"
  getResult(): number {
    return this.result;
  }
}
