describe('integration: api and page mix', () => {
  describe('nested unit block', () => {
    it('does simple assertion', () => {});
  });
  it('hits api', async () => {
    await fetch('http://localhost:3000/api/items');
  });
});
