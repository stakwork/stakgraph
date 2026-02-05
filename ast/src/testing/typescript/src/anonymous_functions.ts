// Basic Arrow Function
const basicArrow = () => {
  console.log("basic");
};

// Arrow Function with Arguments
const arrowWithArgs = (a: number, b: number) => {
  return a + b;
};

// Implicit Return Arrow Function
const implicitReturn = (x: number) => x * 2;

// Anonymous Function Expression
const anonExpr = function () {
  console.log("anon");
};

// Anonymous Function with Args
const anonWithArgs = function (name: string) {
  return "Hello " + name;
};

// Callback Arrow Function
[1, 2, 3].map((x) => x * 2);

// Callback Anonymous Function
[1, 2, 3].forEach(function (x) {
  console.log(x);
});

// IIFE (Immediately Invoked Function Expression)
(() => {
  console.log("IIFE");
})();
