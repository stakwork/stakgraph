describe('e2e: cypress style flow', () => {
  it('visits items', () => {
    cy.visit('http://localhost:3000/items');
    cy.get('input[placeholder="Title"]').type('X');
  });
});
