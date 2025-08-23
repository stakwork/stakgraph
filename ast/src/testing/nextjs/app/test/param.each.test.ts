test.each([[1],[2]])('unit: param %s', (n) => {
  expect(typeof n).toBe('number');
});
