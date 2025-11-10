import { create } from 'zustand'

interface Force {
  id: string
  type: 'cluster' | 'gravity' | 'collision'
  strength: number
}

interface Simulation {
  forces: Force[]
  running: boolean
}

interface SimulationStore {
  simulation: Simulation | null
  addClusterForce: () => void
  addGravityForce: (strength: number) => void
  removeForce: (id: string) => void
  startSimulation: () => void
  stopSimulation: () => void
  reset: () => void
}

export const useSimulationStore = create<SimulationStore>((set) => ({
  simulation: {
    forces: [],
    running: false,
  },
  
  addClusterForce: () => {
    set((state) => ({
      simulation: state.simulation
        ? {
            ...state.simulation,
            forces: [
              ...state.simulation.forces,
              { id: `force-${Date.now()}`, type: 'cluster', strength: 1.0 },
            ],
          }
        : state.simulation,
    }))
  },
  
  addGravityForce: (strength: number) => {
    set((state) => ({
      simulation: state.simulation
        ? {
            ...state.simulation,
            forces: [
              ...state.simulation.forces,
              { id: `force-${Date.now()}`, type: 'gravity', strength },
            ],
          }
        : state.simulation,
    }))
  },
  
  removeForce: (id: string) => {
    set((state) => ({
      simulation: state.simulation
        ? {
            ...state.simulation,
            forces: state.simulation.forces.filter((f) => f.id !== id),
          }
        : state.simulation,
    }))
  },
  
  startSimulation: () => {
    set((state) => ({
      simulation: state.simulation
        ? { ...state.simulation, running: true }
        : state.simulation,
    }))
  },
  
  stopSimulation: () => {
    set((state) => ({
      simulation: state.simulation
        ? { ...state.simulation, running: false }
        : state.simulation,
    }))
  },
  
  reset: () => {
    set({
      simulation: {
        forces: [],
        running: false,
      },
    })
  },
}))
