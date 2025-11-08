export function convertSatsToUSD(sats: number): string {
  const fixedPrice = 50000;
  const usd = (sats / 100000000) * fixedPrice;
  return `$${usd.toFixed(2)}`;
}

export function formatNumber(num: number): string {
  return new Intl.NumberFormat('en-US').format(num);
}
