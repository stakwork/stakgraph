import { describe, it, expect } from 'vitest'
import { convertSatsToUSD } from '@/lib/currency'

describe('unit: currency conversion', () => {
  it('should convert small amounts correctly', () => {
    expect(convertSatsToUSD(1, 50000)).toBe('0.00')
  })

  it('should convert 1 BTC correctly', () => {
    expect(convertSatsToUSD(100000000, 50000)).toBe('50000.00')
  })

  it('should convert 0.1 BTC correctly', () => {
    expect(convertSatsToUSD(10000000, 60000)).toBe('6000.00')
  })
})
