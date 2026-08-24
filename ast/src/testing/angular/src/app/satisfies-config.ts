// Exercises exported variable initializers wrapped in a value expression:
// the `satisfies` operator (TS 5.0) and `as const` / `as` assertions.

// @ast node: Var "featureFlags"
export const featureFlags = {
  auth: true,
  beta: false,
} satisfies Record<string, boolean>;

// @ast node: Var "userRoles"
export const userRoles = ["admin", "editor", "viewer"] as const;
