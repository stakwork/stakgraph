// @ast node: Function "convertSatsToUSD"
// @ast edge: Contains <- File "helpers.ts" "lib/helpers.ts"
export function convertSatsToUSD(sats: number): string {
  const fixedPrice = 50000;
  const usd = (sats / 100000000) * fixedPrice;
  return `$${usd.toFixed(2)}`;
}

// @ast node: Function "formatNumber"
// @ast edge: Contains <- File "helpers.ts" "lib/helpers.ts"
export function formatNumber(num: number): string {
  return new Intl.NumberFormat("en-US").format(num);
}
