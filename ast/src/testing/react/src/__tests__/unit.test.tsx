import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { MemoizedCard } from "../components/HigherOrderComponents";

describe("MemoizedCard Component", () => {
  test("renders title and content", () => {
    render(<MemoizedCard title="Test Title" content="Test Content" />);
    expect(screen.getByText("Test Title")).toBeInTheDocument();
    expect(screen.getByText("Test Content")).toBeInTheDocument();
  });

  it("should apply card className", () => {
    const { container } = render(
      <MemoizedCard title="Title" content="Content" />
    );
    expect(container.firstChild).toHaveClass("card");
  });

  test.skip("skipped test example", () => {
    // This test is skipped
    expect(true).toBe(false);
  });

  test.todo("implement accessibility tests");
});

describe("Form Validation", () => {
  test.only("validates required fields", () => {
    // Only this test runs when using .only
    const mockSubmit = jest.fn();
    expect(mockSubmit).not.toHaveBeenCalled();
  });

  it("validates email format", () => {
    const isValidEmail = (email: string) => email.includes("@");
    expect(isValidEmail("test@example.com")).toBe(true);
    expect(isValidEmail("invalid")).toBe(false);
  });
});

describe("Utility Functions", () => {
  test("formats currency correctly", () => {
    const formatCurrency = (amount: number) => `$${amount.toFixed(2)}`;
    expect(formatCurrency(10)).toBe("$10.00");
    expect(formatCurrency(99.9)).toBe("$99.90");
  });
});
