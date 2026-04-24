// @ts-nocheck
// @ast node: Function "createApiHandlers"
// @ast edge: Contains <- File "noisy-patterns.ts" "lib/noisy-patterns.ts"
// @ast node: Function "onSuccess"
// @ast edge: Contains <- File "noisy-patterns.ts" "lib/noisy-patterns.ts"
// @ast edge: NestedIn -> Function "createApiHandlers" "lib/noisy-patterns.ts"
// @ast node: Function "fetchUser"
// @ast edge: Contains <- File "noisy-patterns.ts" "lib/noisy-patterns.ts"
// @ast node: Function "deleteUser"
// @ast edge: Contains <- File "noisy-patterns.ts" "lib/noisy-patterns.ts"
// @ast node: Var "deepConfig"
// @ast edge: Contains <- File "noisy-patterns.ts" "lib/noisy-patterns.ts"
// @ast node: Function "handler"
// @ast edge: Contains <- File "noisy-patterns.ts" "lib/noisy-patterns.ts"
// @ast edge: NestedIn -> Var "deepConfig" "lib/noisy-patterns.ts"
// @ast node: Function "transform"
// @ast edge: Contains <- File "noisy-patterns.ts" "lib/noisy-patterns.ts"
// @ast edge: NestedIn -> Var "deepConfig" "lib/noisy-patterns.ts"
// @ast node: Class "ApiService"
// @ast edge: Contains <- File "noisy-patterns.ts" "lib/noisy-patterns.ts"
// @ast node: Function "getHandlers"
// @ast edge: Contains <- File "noisy-patterns.ts" "lib/noisy-patterns.ts"
// @ast node: Var "apiService"
// @ast edge: Contains <- File "noisy-patterns.ts" "lib/noisy-patterns.ts"
// @ast absent: Function "text" "lib/noisy-patterns.ts"
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
// @ast node: Function "json"
// @ast node: Var "global.fetch"
