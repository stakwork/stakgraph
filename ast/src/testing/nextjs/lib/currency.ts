// @ast node: Function "convertSatsToUSD"
// @ast edge: Contains <- File "currency.ts" "src/testing/nextjs/lib/currency.ts"
// @ast edge: Calls <- UnitTest "unit: currency conversion" "src/testing/nextjs/app/test/currency.test.ts"
export function convertSatsToUSD(sats: number, btcPrice: number): string {
  const usd = (sats / 100000000) * btcPrice;
  return usd.toFixed(2);
}

// @ast node: Function "convertUSDToSats"
// @ast edge: Contains <- File "currency.ts" "src/testing/nextjs/lib/currency.ts"
export function convertUSDToSats(usd: number, btcPrice: number): number {
  return Math.round((usd / btcPrice) * 100000000);
}
