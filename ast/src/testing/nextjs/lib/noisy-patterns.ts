// @ts-nocheck
import { formatNumber } from "./helpers";

global.fetch = jest.fn().mockResolvedValue({
  json: async () => ({ id: 1, name: "mock" }),
  text: async () => "raw body",
});


jest.mock("./helpers", () => ({
  fetchUser: async (id: string) => ({ id, name: "mocked-user" }),
  deleteUser: async (id: string) => true,
}));

export function createApiHandlers() {
  return {
    onSuccess: (data: any) => formatNumber(data.amount),
    onError: (err: any) => console.error(err),
  };
}

export const deepConfig = {
  level1: {
    level2: {
      handler: async () => {
        return { ready: true };
      },
      transform: (input: string) => input.toUpperCase(),
    },
  },
};

class ApiService {
  getHandlers() {
    return {
      parse: (raw: string) => JSON.parse(raw),
      serialize: (obj: any) => JSON.stringify(obj),
    };
  }
}

export const apiService = new ApiService();
