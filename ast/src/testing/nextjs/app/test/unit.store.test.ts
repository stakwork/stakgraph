// @ts-nocheck
import { useSimulationStore } from "../../lib/stores/simulationStore";

describe("unit: simulation store", () => {
  beforeEach(() => {
    const store = useSimulationStore.getState();
    store.reset();
  });

  it("adds cluster force", () => {
    const store = useSimulationStore.getState();
    const initialLength = store.simulation?.forces.length || 0;

    store.addClusterForce();

    expect(store.simulation?.forces.length).toBe(initialLength + 1);
    const lastForce = store.simulation?.forces[store.simulation.forces.length - 1];
    expect(lastForce?.type).toBe("cluster");
    console.log("Added cluster force:", lastForce);
  });

  it("adds gravity force with custom strength", () => {
    const store = useSimulationStore.getState();
    const strength = 2.5;

    store.addGravityForce(strength);

    const gravityForce = store.simulation?.forces.find(
      (f) => f.type === "gravity"
    );
    expect(gravityForce).toBeDefined();
    expect(gravityForce?.strength).toBe(strength);
    console.log("Added gravity force:", gravityForce);
  });

  it("removes force by id", () => {
    const store = useSimulationStore.getState();
    
    store.addClusterForce();
    const forceId = store.simulation?.forces[0]?.id;
    expect(forceId).toBeDefined();

    store.removeForce(forceId!);

    const foundForce = store.simulation?.forces.find((f) => f.id === forceId);
    expect(foundForce).toBeUndefined();
    console.log("Removed force with id:", forceId);
  });

  it("starts and stops simulation", () => {
    const store = useSimulationStore.getState();

    store.startSimulation();
    expect(store.simulation?.running).toBe(true);

    store.stopSimulation();
    expect(store.simulation?.running).toBe(false);
    console.log("Start/stop simulation works");
  });

  it("resets simulation", () => {
    const store = useSimulationStore.getState();
    
    store.addClusterForce();
    store.addGravityForce(1.0);
    store.startSimulation();

    store.reset();

    expect(store.simulation?.forces.length).toBe(0);
    expect(store.simulation?.running).toBe(false);
    console.log("Reset simulation");
  });
});

describe("unit: store method composition", () => {
  it("chains multiple store operations", () => {
    const store = useSimulationStore.getState();
    store.reset();

    store.addClusterForce();
    store.addGravityForce(1.5);
    store.addGravityForce(2.0);
    store.startSimulation();

    expect(store.simulation?.forces.length).toBe(3);
    expect(store.simulation?.running).toBe(true);

    const clusterCount = store.simulation?.forces.filter(
      (f) => f.type === "cluster"
    ).length;
    const gravityCount = store.simulation?.forces.filter(
      (f) => f.type === "gravity"
    ).length;

    expect(clusterCount).toBe(1);
    expect(gravityCount).toBe(2);
    console.log("Chained operations successful");
  });
});
