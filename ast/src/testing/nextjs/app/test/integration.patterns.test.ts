import { ApiClient } from './helpers/api-client';
import { API_ENDPOINTS, TEST_PRODUCT, TEST_CATEGORY, TEST_COMMENT, TEST_REVIEW } from './fixtures/api-fixtures';

// @ast node: Function "createProduct"
// @ast edge: Contains <- File "integration.patterns.test.ts" "src/testing/nextjs/app/test/integration.patterns.test.ts"
async function createProduct(data: any) {
  return fetch('http://localhost:3000/api/products', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
}

// @ast node: Function "listProducts"
// @ast edge: Contains <- File "integration.patterns.test.ts" "src/testing/nextjs/app/test/integration.patterns.test.ts"
async function listProducts() {
  return fetch('http://localhost:3000/api/products');
}

// @ast node: Function "apiPost"
// @ast edge: Contains <- File "integration.patterns.test.ts" "src/testing/nextjs/app/test/integration.patterns.test.ts"
async function apiPost(path: string, data: any) {
  return fetch(`http://localhost:3000${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
}

// @ast node: Function "createComment"
// @ast edge: Contains <- File "integration.patterns.test.ts" "src/testing/nextjs/app/test/integration.patterns.test.ts"
async function createComment(text: string) {
  return apiPost('/api/comments', { text });
}

// @ast node: Class "ReviewBuilder"
// @ast edge: Contains <- File "integration.patterns.test.ts" "src/testing/nextjs/app/test/integration.patterns.test.ts"
class ReviewBuilder {
  review: any;
  
  constructor() {
    this.review = { rating: 5 };
  }
  
  withRating(rating: number) {
    this.review.rating = rating;
    return this;
  }
  
  withComment(comment: string) {
    this.review.comment = comment;
    return this;
  }
  
  async create() {
    return fetch('http://localhost:3000/api/reviews', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(this.review)
    });
  }
}

// @ast node: IntegrationTest "integration: pattern 1 - helper functions"
// @ast edge: Contains <- File "integration.patterns.test.ts" "src/testing/nextjs/app/test/integration.patterns.test.ts"
describe('integration: pattern 1 - helper functions', () => {
  it('creates product via helper', async () => {
    const res = await createProduct(TEST_PRODUCT);
    expect(res.status).toBe(201);
  });

  it('lists products via helper', async () => {
    const res = await listProducts();
    expect(res.status).toBe(200);
  });
});

describe('integration: pattern 2 - api client class', () => {
  let client: ApiClient;
  
  beforeAll(() => {
    client = new ApiClient();
  });

  it('updates user via client', async () => {
    const res = await client.put('/api/users/1', { name: 'Updated User' });
    expect(res.status).toBe(200);
  });

  it('deletes user via client', async () => {
    const res = await client.delete('/api/users/1');
    expect(res.status).toBe(200);
  });
});

describe('integration: pattern 3 - fixtures and constants', () => {
  it('creates category with fixture', async () => {
    const res = await fetch(`http://localhost:3000${API_ENDPOINTS.categories}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(TEST_CATEGORY)
    });
    expect(res.status).toBe(201);
  });

  it('gets category by id with fixture', async () => {
    const res = await fetch(`http://localhost:3000${API_ENDPOINTS.categoryById('1')}`);
    expect(res.status).toBe(200);
  });
});

describe('integration: pattern 4 - setup and teardown', () => {
  let sessionId: string;
  
  beforeEach(async () => {
    const res = await fetch('http://localhost:3000/api/sessions', {
      method: 'POST'
    });
    const data = await res.json();
    sessionId = data.id;
  });

  afterEach(async () => {
    await fetch(`http://localhost:3000/api/sessions/${sessionId}`, {
      method: 'DELETE'
    });
  });

  it('uses session from setup', () => {
    expect(sessionId).toBeDefined();
  });
});

describe('integration: pattern 5 - nested helpers', () => {
  it('creates comment via nested helpers', async () => {
    const res = await createComment('Great post!');
    expect(res.status).toBe(201);
  });

  it('lists comments', async () => {
    const res = await fetch('http://localhost:3000/api/comments');
    expect(res.status).toBe(200);
  });
});

describe('integration: pattern 6 - builder pattern', () => {
  it('creates review via builder', async () => {
    const res = await new ReviewBuilder().withRating(4).create();
    expect(res.status).toBe(201);
  });

  it('updates review', async () => {
    const res = await fetch('http://localhost:3000/api/reviews/1', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rating: 5 })
    });
    expect(res.status).toBe(200);
  });
});
