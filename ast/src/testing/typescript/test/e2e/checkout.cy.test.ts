// @ts-nocheck

describe("e2e: checkout flow", () => {
  it("completes checkout process", () => {
    cy.visit("/checkout");
    cy.get('input[name="card"]').type("4111111111111111");
    cy.get('button[type="submit"]').click();
    cy.contains("Order confirmed").should("be.visible");
  });

  it("handles payment errors", () => {
    cy.visit("/checkout");
    cy.get('input[name="card"]').type("0000000000000000");
    cy.get('button[type="submit"]').click();
    cy.contains("Payment failed").should("be.visible");
  });
});
