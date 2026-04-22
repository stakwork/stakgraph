// @ast node: Function "format"
// @ast edge: Contains <- File "api-handlers.ts" "src/testing/nextjs/lib/api-handlers.ts"
// @ast node: Function "display"
// @ast edge: Contains <- File "api-handlers.ts" "src/testing/nextjs/lib/api-handlers.ts"
import { convertSatsToUSD } from "./currency";
import { formatNumber } from "./helpers";

export const bountyHandlers = {
  format: (sats: number) => convertSatsToUSD(sats),
  display: (num: number) => formatNumber(num),
};
