describe('integration: items and person apis', () => {
  beforeAll(async () => {});
  it('fetches items', async () => {
    await fetch('http://localhost:3000/api/items');
  });
  it('creates person', async () => {
    await fetch('http://localhost:3000/api/person', { method: 'POST' });
  });
});
