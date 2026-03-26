import { memo, useEffect, useCallback } from "react";
import { Canvas } from "@react-three/fiber";
import { CameraControls, AdaptiveDpr, Preload } from "@react-three/drei";
import { Graph } from "./components/Graph";
import { useGraphData } from "@/stores/useGraphData";
import { useSimulation } from "@/stores/useSimulation";
import type { GraphApiResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE || "";

const SceneContent = memo(() => {
  return (
    <>
      <CameraControls
        makeDefault
        minDistance={50}
        maxDistance={15000}
        smoothTime={0.8}
        dollyToCursor
      />
      <Graph />
    </>
  );
});

SceneContent.displayName = "SceneContent";

export const GraphScene = memo(() => {
  const setData = useGraphData((s) => s.setData);
  const data = useGraphData((s) => s.data);
  const nodeTypes = useGraphData((s) => s.nodeTypes);
  const loading = useGraphData((s) => s.loading);
  const layoutNodes = useSimulation((s) => s.layoutNodes);
  const destroy = useSimulation((s) => s.destroy);

  const fetchGraph = useCallback(async () => {
    try {
      const [codeRes, featuresRes] = await Promise.all([
        fetch(
          `${API_BASE}/graph?edges=true&no_body=true&limit=500&limit_mode=per_type`
        ),
        fetch(`${API_BASE}/gitree/all-features-graph?no_body=true`),
      ]);

      const codeData: GraphApiResponse = codeRes.ok
        ? await codeRes.json()
        : { nodes: [], edges: [], status: "error" };
      const featureData: GraphApiResponse = featuresRes.ok
        ? await featuresRes.json()
        : { nodes: [], edges: [], status: "error" };

      // Merge, dedup nodes by ref_id
      const nodeMap = new Map<string, (typeof codeData.nodes)[0]>();
      for (const n of [...codeData.nodes, ...featureData.nodes]) {
        if (!nodeMap.has(n.ref_id)) nodeMap.set(n.ref_id, n);
      }

      const allEdges = [...codeData.edges, ...featureData.edges];

      setData(Array.from(nodeMap.values()), allEdges);
    } catch (err) {
      console.error("Failed to fetch graph data:", err);
    }
  }, [setData]);

  useEffect(() => {
    fetchGraph();
    return () => destroy();
  }, [fetchGraph, destroy]);

  // Layout nodes when data arrives
  useEffect(() => {
    if (data && data.nodes.length > 0) {
      layoutNodes(data.nodes, nodeTypes);
    }
  }, [data, nodeTypes, layoutNodes]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        Loading graph...
      </div>
    );
  }

  if (!data || data.nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        No graph data available
      </div>
    );
  }

  return (
    <div style={{ width: "100%", height: "100%" }}>
      <Canvas
        camera={{
          position: [2000, 2000, 4000],
          far: 30000,
          near: 1,
        }}
        style={{ width: "100%", height: "100%" }}
      >
        <AdaptiveDpr />
        <Preload />
        <SceneContent />
      </Canvas>
    </div>
  );
});

GraphScene.displayName = "GraphScene";
