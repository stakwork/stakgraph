import { create } from 'zustand'

interface CountStore {
  count: number
  increment: () => void
  decrement: () => void
  reset: () => void
  getCount: () => number
}

export const useCountStore = create<CountStore>((set, get) => ({
  count: 0,
  
  increment: () => {
    set((state) => ({ count: state.count + 1 }))
  },
  
  decrement: () => {
    set((state) => ({ count: state.count - 1 }))
  },
  
  reset: () => {
    set({ count: 0 })
  },
  
  getCount: () => {
    return get().count
  },
}))
