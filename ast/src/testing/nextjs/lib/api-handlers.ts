import { convertSatsToUSD } from "./currency";
import { formatNumber } from "./helpers";

export const bountyHandlers = {
  format: (sats: number) => convertSatsToUSD(sats),
  display: (num: number) => formatNumber(num),
};
