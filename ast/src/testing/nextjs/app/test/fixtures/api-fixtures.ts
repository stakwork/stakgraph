export const API_BASE_URL = 'http://localhost:3000';

export const API_ENDPOINTS = {
  products: '/api/products',
  users: (id: string) => `/api/users/${id}`,
  categories: '/api/categories',
  categoryById: (id: string) => `/api/categories/${id}`,
  sessions: '/api/sessions',
  sessionById: (id: string) => `/api/sessions/${id}`,
  comments: '/api/comments',
  reviews: '/api/reviews',
  reviewById: (id: string) => `/api/reviews/${id}`
};

export const TEST_PRODUCT = {
  name: 'Test Widget',
  price: 29.99
};

export const TEST_CATEGORY = {
  name: 'Electronics'
};

export const TEST_COMMENT = {
  text: 'Great post!'
};

export const TEST_REVIEW = {
  rating: 5,
  comment: 'Excellent!'
};
