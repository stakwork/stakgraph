import { useMemo, useRef, useState } from "react";
import { Html } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { Plane, Raycaster, Vector2, Vector3 } from "three";
import { useGraphData, getColorForType } from "@/stores/useGraphData";
import { nodePositions } from "@/stores/useSimulation";
import { LAYER_SPACING, GRID_PADDING } from "@/graph/config";

const HOVER_THRESHOLD = 200;

const EDGE_COLOR = "#fff";
const EDGE_OPACITY = 0.5;
const FILL_COLOR = "#fff";
const FILL_OPACITY = 0.06;

type LayerInfo = {
  nodeType: string;
  yPosition: number;
};

type Bounds = {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
};

function calculateLayerBounds(nodeType: string): Bounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  let found = false;

  const data = useGraphData.getState().data;
  if (!data) return { minX: -500, maxX: 500, minZ: -300, maxZ: 300 };

  for (const node of data.nodes) {
    if (node.node_type !== nodeType) continue;
    const pos = nodePositions.get(node.ref_id);
    if (!pos) continue;
    found = true;
    if (pos.x < minX) minX = pos.x;
    if (pos.x > maxX) maxX = pos.x;
    if (pos.z < minZ) minZ = pos.z;
    if (pos.z > maxZ) maxZ = pos.z;
  }

  if (!found) return { minX: -500, maxX: 500, minZ: -300, maxZ: 300 };

  return {
    minX: minX - GRID_PADDING,
    maxX: maxX + GRID_PADDING,
    minZ: minZ - GRID_PADDING,
    maxZ: maxZ + GRID_PADDING,
  };
}

export const LayerHoverHighlight = () => {
  const nodeTypes = useGraphData((s) => s.nodeTypes);
  const selectedNode = useGraphData((s) => s.selectedNode);

  const [hoveredLayer, setHoveredLayer] = useState<LayerInfo | null>(null);
  const [bounds, setBounds] = useState<Bounds | null>(null);

  const { camera } = useThree();
  const mouseRef = useRef(new Vector2());
  const raycaster = useRef(new Raycaster());
  const intersectPlane = useRef(new Plane(new Vector3(0, 0, 1), 0));
  const intersectPoint = useRef(new Vector3());
  const lastBoundsRef = useRef("");

  const layerPositions = useMemo(() => {
    const totalTypes = nodeTypes.length;
    const startOffset = ((totalTypes - 1) / 2) * LAYER_SPACING;

    return nodeTypes.map((nodeType, index) => ({
      nodeType,
      yPosition: startOffset - index * LAYER_SPACING,
    }));
  }, [nodeTypes]);

  useFrame(({ mouse }) => {
    if (selectedNode) {
      if (hoveredLayer) {
        setHoveredLayer(null);
        setBounds(null);
        lastBoundsRef.current = "";
      }
      return;
    }

    mouseRef.current.set(mouse.x, mouse.y);
    raycaster.current.setFromCamera(mouseRef.current, camera);

    const hasIntersection = raycaster.current.ray.intersectPlane(
      intersectPlane.current,
      intersectPoint.current
    );

    if (!hasIntersection) {
      if (hoveredLayer) {
        setHoveredLayer(null);
        setBounds(null);
        lastBoundsRef.current = "";
      }
      return;
    }

    const worldY = intersectPoint.current.y;
    const worldX = intersectPoint.current.x;
    const worldZ = intersectPoint.current.z;

    let closestLayer: LayerInfo | null = null;
    let minDistance = HOVER_THRESHOLD;

    for (const layer of layerPositions) {
      const distance = Math.abs(worldY - layer.yPosition);
      if (distance < minDistance) {
        minDistance = distance;
        closestLayer = layer;
      }
    }

    if (!closestLayer) {
      if (hoveredLayer) {
        setHoveredLayer(null);
        setBounds(null);
        lastBoundsRef.current = "";
      }
      return;
    }

    const currentBounds = calculateLayerBounds(closestLayer.nodeType);
    const boundsKey = `${Math.round(currentBounds.minX)}-${Math.round(currentBounds.maxX)}-${Math.round(currentBounds.minZ)}-${Math.round(currentBounds.maxZ)}`;

    const isWithinBounds =
      worldX >= currentBounds.minX &&
      worldX <= currentBounds.maxX &&
      worldZ >= currentBounds.minZ &&
      worldZ <= currentBounds.maxZ;

    if (!isWithinBounds) {
      if (hoveredLayer) {
        setHoveredLayer(null);
        setBounds(null);
        lastBoundsRef.current = "";
      }
      return;
    }

    if (closestLayer.nodeType !== hoveredLayer?.nodeType) {
      setHoveredLayer(closestLayer);
      setBounds(currentBounds);
      lastBoundsRef.current = boundsKey;
    } else if (boundsKey !== lastBoundsRef.current) {
      setBounds(currentBounds);
      lastBoundsRef.current = boundsKey;
    }
  });

  if (!hoveredLayer || !bounds || selectedNode) return null;

  const { nodeType, yPosition } = hoveredLayer;
  const { minX, maxX, minZ, maxZ } = bounds;
  const width = maxX - minX;
  const depth = maxZ - minZ;
  const centerX = (minX + maxX) / 2;
  const centerZ = (minZ + maxZ) / 2;
  const color = getColorForType(nodeType);

  const boundsKey = `${nodeType}-${Math.round(minX)}-${Math.round(maxX)}-${Math.round(minZ)}-${Math.round(maxZ)}`;

  return (
    <group name="layer-hover-highlight" key={boundsKey}>
      {/* Transparent fill */}
      <mesh
        position={[centerX, yPosition, centerZ]}
        rotation={[-Math.PI / 2, 0, 0]}
      >
        <planeGeometry args={[width, depth]} />
        <meshBasicMaterial
          color={FILL_COLOR}
          transparent
          opacity={FILL_OPACITY}
          depthWrite={false}
          side={2}
        />
      </mesh>

      {/* Edge lines */}
      <lineLoop position={[centerX, yPosition, centerZ]}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[
              new Float32Array([
                -width / 2, 0, -depth / 2,
                width / 2, 0, -depth / 2,
                width / 2, 0, depth / 2,
                -width / 2, 0, depth / 2,
              ]),
              3,
            ]}
          />
        </bufferGeometry>
        <lineBasicMaterial
          color={EDGE_COLOR}
          transparent
          opacity={EDGE_OPACITY}
        />
      </lineLoop>

      {/* Label */}
      <Html
        position={[maxX + 80, yPosition, centerZ]}
        center
        zIndexRange={[100, 101]}
        sprite={false}
        style={{ pointerEvents: "none", userSelect: "none" }}
      >
        <div
          style={{
            background: color,
            color: "white",
            padding: "6px 14px",
            borderRadius: 8,
            fontSize: 13,
            fontWeight: 600,
            whiteSpace: "nowrap",
            fontFamily: "system-ui, sans-serif",
          }}
        >
          {nodeType}
        </div>
      </Html>
    </group>
  );
};
