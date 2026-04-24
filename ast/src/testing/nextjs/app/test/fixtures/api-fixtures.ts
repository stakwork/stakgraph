// @ast node: Var "API_BASE_URL"
// @ast edge: Contains <- File "api-fixtures.ts" "src/testing/nextjs/app/test/fixtures/api-fixtures.ts"
export const API_BASE_URL = "http://localhost:3000";

// @ast node: Var "API_ENDPOINTS"
// @ast edge: Contains <- File "api-fixtures.ts" "src/testing/nextjs/app/test/fixtures/api-fixtures.ts"
export const API_ENDPOINTS = {
  products: "/api/products",
  users: (id: string) => `/api/users/${id}`,
  categories: "/api/categories",
  categoryById: (id: string) => `/api/categories/${id}`,
  sessions: "/api/sessions",
  sessionById: (id: string) => `/api/sessions/${id}`,
  comments: "/api/comments",
  reviews: "/api/reviews",
  reviewById: (id: string) => `/api/reviews/${id}`,
};

// @ast node: Var "TEST_PRODUCT"
// @ast edge: Contains <- File "api-fixtures.ts" "src/testing/nextjs/app/test/fixtures/api-fixtures.ts"
export const TEST_PRODUCT = {
  name: "Test Widget",
  price: 29.99,
};

// @ast node: Var "TEST_CATEGORY"
// @ast edge: Contains <- File "api-fixtures.ts" "src/testing/nextjs/app/test/fixtures/api-fixtures.ts"
export const TEST_CATEGORY = {
  name: "Electronics",
};

// @ast node: Var "TEST_COMMENT"
// @ast edge: Contains <- File "api-fixtures.ts" "src/testing/nextjs/app/test/fixtures/api-fixtures.ts"
export const TEST_COMMENT = {
  text: "Great post!",
};

// @ast node: Var "TEST_REVIEW"
// @ast edge: Contains <- File "api-fixtures.ts" "src/testing/nextjs/app/test/fixtures/api-fixtures.ts"
export const TEST_REVIEW = {
  rating: 5,
  comment: "Excellent!",
};
// @ast node: Function "categoryById"
