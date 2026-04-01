import { memo, useEffect, useCallback, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { CameraControls, AdaptiveDpr, Preload } from "@react-three/drei";
import { MathUtils } from "three";
import { Graph } from "./components/Graph";
import { useGraphData } from "@/stores/useGraphData";
import { nodePositions, useSimulation } from "@/stores/useSimulation";
import { useIngestion } from "@/stores/useIngestion";
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
const ZOOM_DISTANCE = 400;

const SceneContent = memo(() => {
  const controlsRef = useRef<CameraControlsImpl>(null);
  const autoRotate = useRef(true);
  const selectedNode = useGraphData((s) => s.selectedNode);
  const setSelectedNode = useGraphData((s) => s.setSelectedNode);
  const prevSelectedRef = useRef<string | null>(null);

  useFrame((_, delta) => {
    if (controlsRef.current && autoRotate.current) {
      controlsRef.current.azimuthAngle += AUTO_ROTATE_SPEED * delta * MathUtils.DEG2RAD;
    }
  });

  // Zoom to selected node
  useEffect(() => {
    const refId = selectedNode?.ref_id ?? null;
    if (refId === prevSelectedRef.current) return;
    prevSelectedRef.current = refId;

    if (!refId || !controlsRef.current) return;
    const pos = nodePositions.get(refId);
    if (!pos) return;

    autoRotate.current = false;
    controlsRef.current.setLookAt(
      pos.x + ZOOM_DISTANCE * 0.3, pos.y + ZOOM_DISTANCE * 0.15, pos.z + ZOOM_DISTANCE,
      pos.x, pos.y, pos.z,
      true,
    );
  }, [selectedNode]);

  // Escape key to deselect
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && useGraphData.getState().selectedNode) {
        setSelectedNode(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setSelectedNode]);

  return (
    <>
      <CameraControls
        ref={controlsRef}
        makeDefault
        minDistance={CAMERA_MIN_DISTANCE}
        maxDistance={CAMERA_MAX_DISTANCE}
        smoothTime={CAMERA_SMOOTH_TIME}
        dollyToCursor
        onStart={() => { autoRotate.current = false; }}
      />
      <Graph />
    </>
  );
});

SceneContent.displayName = "SceneContent";

export const GraphScene = memo(() => {
  const setData = useGraphData((s) => s.setData);
  const addNodes = useGraphData((s) => s.addNodes);
  const data = useGraphData((s) => s.data);
  const nodeTypes = useGraphData((s) => s.nodeTypes);
  const loading = useGraphData((s) => s.loading);
  const layoutNodes = useSimulation((s) => s.layoutNodes);
  const layoutNodesIncremental = useSimulation((s) => s.layoutNodesIncremental);
  const destroy = useSimulation((s) => s.destroy);
  const hasInitialData = useRef(false);
  const isFetchingRef = useRef(false);
  const latestTimestampRef = useRef<number | null>(null);

  const fetchGraph = useCallback(
    async (incremental = false) => {
      if (isFetchingRef.current) return;
      isFetchingRef.current = true;
      try {
        const nodeTypesParam = LAYER_ORDER.join(",");
        const sinceParam =
          incremental && latestTimestampRef.current !== null
            ? `&since=${latestTimestampRef.current}`
            : "";

        const [codeRes, featuresRes] = await Promise.all([
          fetch(
            `${API_BASE}/graph?edges=true&no_body=true&limit=500&limit_mode=per_type&node_types=${nodeTypesParam}${sinceParam}`,
          ),
          fetch(
            `${API_BASE}/gitree/all-features-graph?no_body=true&node_types=${nodeTypesParam}`,
          ),
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

        const allNodes = Array.from(nodeMap.values());
        const allEdges = [...codeData.edges, ...featureData.edges];

        // Track latest timestamp for delta fetching
        for (const n of allNodes) {
          const ts = n.date_added_to_graph as number | undefined;
          if (
            ts &&
            (latestTimestampRef.current === null ||
              ts > latestTimestampRef.current)
          ) {
            latestTimestampRef.current = ts;
          }
        }

        if (incremental) {
          addNodes(allNodes, allEdges);
        } else {
          setData(allNodes, allEdges);
        }
      } catch (err) {
        console.error("Failed to fetch graph data:", err);
      } finally {
        isFetchingRef.current = false;
      }
    },
    [setData, addNodes],
  );

  useEffect(() => {
    fetchGraph();
    return () => {
      destroy();
      hasInitialData.current = false;
    };
  }, [fetchGraph, destroy]);

  const statsVersion = useIngestion((s) => s.statsVersion);
  const ingestionPhase = useIngestion((s) => s.phase);
  useEffect(() => {
    if (ingestionPhase !== "running" || statsVersion === 0) return;
    fetchGraph(true);
  }, [statsVersion, ingestionPhase, fetchGraph]);

  useEffect(() => {
    if (ingestionPhase === "complete") {
      // Full re-fetch after ingestion: clear stale incremental state so we
      // load all nodes (including ones from the first repo) rather than only
      // nodes added since the last delta timestamp.
      latestTimestampRef.current = null;
      hasInitialData.current = false;
      fetchGraph(false);
    }
  }, [ingestionPhase, fetchGraph]);

  const prevNodeTypesRef = useRef<string[]>([]);

  // Layout nodes when data arrives.
  // First load or new node types appearing: full layout (clears and recomputes
  // everything so labels and node positions stay in sync).
  // Subsequent polls with the same type set: incremental layout only.
  useEffect(() => {
    if (!data || data.nodes.length === 0) return;
    const typesChanged =
      nodeTypes.length !== prevNodeTypesRef.current.length ||
      nodeTypes.some((t, i) => t !== prevNodeTypesRef.current[i]);
    prevNodeTypesRef.current = nodeTypes;

    if (!hasInitialData.current || typesChanged) {
      hasInitialData.current = true;
      layoutNodes(data.nodes, nodeTypes);
    } else {
      layoutNodesIncremental(data.nodes, nodeTypes);
    }
  }, [data, nodeTypes, layoutNodes, layoutNodesIncremental]);

  const isEmpty = !loading && (!data || data.nodes.length === 0);
  const isWaiting = isEmpty && ingestionPhase === "running";

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        Loading graph...
      </div>
    );
  }

  if (isWaiting) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3 text-muted-foreground">
        <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
        <span className="text-sm">Graph Loading…</span>
      </div>
    );
  }

  if (isEmpty) {
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
