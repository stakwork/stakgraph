import axios from "axios";
import ky from "ky";

const API_BASE = "http://localhost:5002";

// Axios patterns
export async function fetchUsersAxios() {
  const response = await axios.get(`${API_BASE}/api/users`);
  return response.data;
}

export async function createUserAxios(name: string, email: string) {
  const response = await axios.post(`${API_BASE}/api/users`, { name, email });
  return response.data;
}

export async function updateUserAxios(id: number, data: object) {
  const response = await axios.put(`${API_BASE}/api/users/${id}`, data);
  return response.data;
}

export async function deleteUserAxios(id: number) {
  const response = await axios.delete(`${API_BASE}/api/users/${id}`);
  return response.data;
}

export async function patchUserAxios(id: number, data: object) {
  const response = await axios.patch(`${API_BASE}/api/users/${id}`, data);
  return response.data;
}

// Axios with config object
export async function fetchWithConfig() {
  const response = await axios({
    url: `${API_BASE}/api/config`,
    method: "GET",
  });
  return response.data;
}

// Ky patterns
export async function fetchUsersKy() {
  const response = await ky.get(`${API_BASE}/api/users`).json();
  return response;
}

export async function createUserKy(name: string, email: string) {
  const response = await ky
    .post(`${API_BASE}/api/users`, {
      json: { name, email },
    })
    .json();
  return response;
}

// Request constructor patterns
export async function fetchWithRequest() {
  const request = new Request(`${API_BASE}/api/data`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key: "value" }),
  });
  const response = await fetch(request);
  return response.json();
}

// NextRequest pattern (for middleware/edge)
export function createNextRequest(path: string) {
  const request = new NextRequest(`${API_BASE}${path}`);
  return request;
}
