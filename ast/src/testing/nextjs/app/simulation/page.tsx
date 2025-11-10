"use client";
import { useEffect, useState } from "react";
import { useSimulationStore } from "../../lib/stores/simulationStore";
import {
  Card,
  CardHeader,
  CardContent,
  CardTitle,
} from "../../components/ui/card";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";

function Simulation() {
  const [forceStrength, setForceStrength] = useState(0.5);
  const store = useSimulationStore();

  useEffect(() => {
    const state = useSimulationStore.getState();
    if (state.simulation && state.simulation.forces.length === 0) {
      state.addClusterForce();
    }
  }, []);

  const handleAddCluster = () => {
    const state = useSimulationStore.getState();
    state.addClusterForce();
  };

  const handleAddGravity = () => {
    const state = useSimulationStore.getState();
    state.addGravityForce(forceStrength);
  };

  const handleRemoveForce = (id: string) => {
    const state = useSimulationStore.getState();
    state.removeForce(id);
  };

  const handleStart = () => {
    const state = useSimulationStore.getState();
    state.startSimulation();
  };

  const handleStop = () => {
    const state = useSimulationStore.getState();
    state.stopSimulation();
  };

  const handleReset = () => {
    const state = useSimulationStore.getState();
    state.reset();
  };

  return (
    <main className="max-w-2xl mx-auto py-8">
      <Card>
        <CardHeader>
          <CardTitle>Physics Simulation</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex gap-2">
              <Button onClick={handleAddCluster}>Add Cluster Force</Button>
              <Button onClick={handleStart}>Start</Button>
              <Button onClick={handleStop}>Stop</Button>
              <Button onClick={handleReset}>Reset</Button>
            </div>

            <div className="flex gap-2 items-center">
              <Input
                type="number"
                step="0.1"
                value={forceStrength}
                onChange={(e) => setForceStrength(Number(e.target.value))}
                className="w-32"
              />
              <Button onClick={handleAddGravity}>Add Gravity Force</Button>
            </div>

            <div className="mt-6">
              <h3 className="font-semibold mb-2">Active Forces:</h3>
              {store.simulation?.forces.length === 0 ? (
                <p className="text-gray-500">No forces active</p>
              ) : (
                <ul className="space-y-2">
                  {store.simulation?.forces.map((force) => (
                    <li
                      key={force.id}
                      className="flex justify-between items-center"
                    >
                      <span>
                        {force.type} (strength: {force.strength})
                      </span>
                      <Button
                        onClick={() => handleRemoveForce(force.id)}
                        variant="destructive"
                        size="sm"
                      >
                        Remove
                      </Button>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="mt-4">
              <p>
                Status:{" "}
                {store.simulation?.running ? (
                  <span className="text-green-600">Running</span>
                ) : (
                  <span className="text-gray-500">Stopped</span>
                )}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </main>
  );
}

export { Simulation as default };
