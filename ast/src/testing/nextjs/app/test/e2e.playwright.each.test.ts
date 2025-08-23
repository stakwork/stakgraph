describe.each([['items'], ['person']])('e2e: %s page flow', (name) => {
  test('navigates ' + name, async () => {
    await page.goto('http://localhost:3000/' + name);
  });
});
