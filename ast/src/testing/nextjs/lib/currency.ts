export function convertSatsToUSD(sats: number, btcPrice: number): string {
  const usd = (sats / 100000000) * btcPrice;
  return usd.toFixed(2);
}

export function convertUSDToSats(usd: number, btcPrice: number): number {
  return Math.round((usd / btcPrice) * 100000000);
}
