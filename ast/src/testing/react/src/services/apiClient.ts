import axios from "axios";
import ky from "ky";

const API_BASE = "http://localhost:5002";
// @ast node: Var "API_BASE"

// Axios patterns
// @ast node: Function "fetchUsersAxios"
export async function fetchUsersAxios() {
  // @ast node: Request "${API_BASE}/api/users" [verb=GET]
  const response = await axios.get(`${API_BASE}/api/users`);
  return response.data;
}

// @ast node: Function "createUserAxios"
export async function createUserAxios(name: string, email: string) {
  // @ast node: Request "${API_BASE}/api/users" [verb=POST]
  const response = await axios.post(`${API_BASE}/api/users`, { name, email });
  return response.data;
}

// @ast node: Function "updateUserAxios"
export async function updateUserAxios(id: number, data: object) {
  // @ast node: Request "${API_BASE}/api/users/${id}" [verb=PUT]
  const response = await axios.put(`${API_BASE}/api/users/${id}`, data);
  return response.data;
}

// @ast node: Function "deleteUserAxios"
export async function deleteUserAxios(id: number) {
  // @ast node: Request "${API_BASE}/api/users/${id}" [verb=DELETE]
  const response = await axios.delete(`${API_BASE}/api/users/${id}`);
  return response.data;
}

// @ast node: Function "patchUserAxios"
export async function patchUserAxios(id: number, data: object) {
  // @ast node: Request "${API_BASE}/api/users/${id}" [verb=PATCH]
  const response = await axios.patch(`${API_BASE}/api/users/${id}`, data);
  return response.data;
}

// Axios with config object
// @ast node: Function "fetchWithConfig"
export async function fetchWithConfig() {
  // @ast node: Request "${API_BASE}/api/config" [verb=GET]
  const response = await axios({
    url: `${API_BASE}/api/config`,
    method: "GET",
  });
  return response.data;
}

// Ky patterns
// @ast node: Function "fetchUsersKy"
export async function fetchUsersKy() {
  // @ast node: Request "${API_BASE}/api/users" [verb=GET]
  const response = await ky.get(`${API_BASE}/api/users`).json();
  return response;
}

// @ast node: Function "createUserKy"
export async function createUserKy(name: string, email: string) {
  // @ast node: Request "${API_BASE}/api/users" [verb=POST]
  const response = await ky
    .post(`${API_BASE}/api/users`, {
      json: { name, email },
    })
    .json();
  return response;
}

// Request constructor patterns
// @ast node: Function "fetchWithRequest"
export async function fetchWithRequest() {
  // @ast node: Request "${API_BASE}/api/data"
  const request = new Request(`${API_BASE}/api/data`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key: "value" }),
  });
  const response = await fetch(request);
  return response.json();
}

// NextRequest pattern (for middleware/edge)
// @ast node: Function "createNextRequest"
export function createNextRequest(path: string) {
  // @ast node: Request "${API_BASE}${path}"
  const request = new NextRequest(`${API_BASE}${path}`);
  return request;
}
