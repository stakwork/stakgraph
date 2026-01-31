import { API_BASE_URL } from "../api/endpoints.js";

/**
 * Handles API interactions
 */
export class ApiClient {
  constructor(baseUrl = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async get(endpoint) {
    try {
      // simulating API call for testing
      if (endpoint === "/updates") {
        return this.simulatedUpdates();
      }
      if (endpoint === "/team") {
        return this.simulatedTeam();
      }

      const response = await fetch(`${this.baseUrl}${endpoint}`);
      if (!response.ok) throw new Error("Network response was not ok");
      return await response.json();
    } catch (error) {
      console.error("API Error:", error);
      throw error;
    }
  }

  async post(endpoint, data) {
    // simulate post
    console.log(`POST to ${endpoint}`, data);
    return { success: true, message: "Data received" };
  }

  simulatedUpdates() {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve([
          { id: 1, title: "Version 1.0 Released", date: new Date() },
          {
            id: 2,
            title: "New Feature: Dark Mode",
            date: new Date(Date.now() - 86400000),
          },
        ]);
      }, 500);
    });
  }

  simulatedTeam() {
    return Promise.resolve([
      { name: "Alice", role: "Frontend" },
      { name: "Bob", role: "Backend" },
      { name: "Charlie", role: "Designer" },
    ]);
  }
}
