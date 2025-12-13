import { describe, it, expect, beforeEach } from 'vitest';
import { useCountStore } from '@/lib/stores/useCountStore';

describe('unit: zustand count store', () => {
  beforeEach(() => {
    useCountStore.getState().reset();
  });

  it('initializes with zero count', () => {
    const state = useCountStore.getState();
    expect(state.count).toBe(0);
  });

  it('increments count', () => {
    const initialCount = useCountStore.getState().getCount();
    
    useCountStore.getState().increment();
    
    const newCount = useCountStore.getState().getCount();
    expect(newCount).toBe(initialCount + 1);
  });

  it('decrements count', () => {
    useCountStore.getState().increment();
    useCountStore.getState().increment();
    const beforeDecrement = useCountStore.getState().getCount();
    
    useCountStore.getState().decrement();
    
    const afterDecrement = useCountStore.getState().getCount();
    expect(afterDecrement).toBe(beforeDecrement - 1);
  });

  it('resets count to zero', () => {
    useCountStore.getState().increment();
    useCountStore.getState().increment();
    expect(useCountStore.getState().count).toBeGreaterThan(0);
    
    useCountStore.getState().reset();
    
    expect(useCountStore.getState().count).toBe(0);
  });

  it('handles multiple operations', () => {
    useCountStore.getState().increment();
    useCountStore.getState().increment();
    useCountStore.getState().decrement();
    
    const finalCount = useCountStore.getState().getCount();
    expect(finalCount).toBe(1);
  });
});
