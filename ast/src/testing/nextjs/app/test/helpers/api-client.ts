// @ast node: Class "ApiClient"
// @ast edge: Contains <- File "api-client.ts" "src/testing/nextjs/app/test/helpers/api-client.ts"
export class ApiClient {
  async get(endpoint: string) {
    return fetch(`http://localhost:3000${endpoint}`);
  }

  async post(endpoint: string, data: any) {
    return fetch(`http://localhost:3000${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
  }

  async put(endpoint: string, data: any) {
    return fetch(`http://localhost:3000${endpoint}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
  }

  async delete(endpoint: string) {
    return fetch(`http://localhost:3000${endpoint}`, {
      method: "DELETE",
    });
  }
}
