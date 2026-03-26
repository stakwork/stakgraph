import { memo, useEffect, useCallback, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { CameraControls, AdaptiveDpr, Preload } from "@react-three/drei";
import { MathUtils } from "three";
import { Graph } from "./components/Graph";
import { useGraphData } from "@/stores/useGraphData";
import { useSimulation } from "@/stores/useSimulation";
import type { GraphApiResponse } from "./types";
import type CameraControlsImpl from "camera-controls";
import {
  LAYER_ORDER,
  INITIAL_CAMERA_POSITION,
  CAMERA_NEAR,
  CAMERA_FAR,
  CAMERA_MIN_DISTANCE,
  CAMERA_MAX_DISTANCE,
  CAMERA_SMOOTH_TIME,
  AUTO_ROTATE_SPEED,
} from "./config";

const API_BASE = import.meta.env.VITE_API_BASE || "";

const SceneContent = memo(() => {
  const controlsRef = useRef<CameraControlsImpl>(null);
  const userInteracting = useRef(false);

  useFrame((_, delta) => {
    if (controlsRef.current && !userInteracting.current) {
      controlsRef.current.azimuthAngle += AUTO_ROTATE_SPEED * delta * MathUtils.DEG2RAD;
    }
  });

  return (
    <>
      <CameraControls
        ref={controlsRef}
        makeDefault
        minDistance={CAMERA_MIN_DISTANCE}
        maxDistance={CAMERA_MAX_DISTANCE}
        smoothTime={CAMERA_SMOOTH_TIME}
        dollyToCursor
        onStart={() => { userInteracting.current = true; }}
        onEnd={() => { userInteracting.current = false; }}
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
      const nodeTypesParam = LAYER_ORDER.join(",");
      const [codeRes, featuresRes] = await Promise.all([
        fetch(
          `${API_BASE}/graph?edges=true&no_body=true&limit=500&limit_mode=per_type&node_types=${nodeTypesParam}`
        ),
        fetch(`${API_BASE}/gitree/all-features-graph?no_body=true&node_types=${nodeTypesParam}`),
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
          position: INITIAL_CAMERA_POSITION,
          far: CAMERA_FAR,
          near: CAMERA_NEAR,
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
