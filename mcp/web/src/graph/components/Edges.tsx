import { memo, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import {
  BufferGeometry,
  Float32BufferAttribute,
  LineBasicMaterial,
  Color,
} from "three";
import { useGraphData } from "@/stores/useGraphData";
import { nodePositions, useSimulation } from "@/stores/useSimulation";
import { EDGE_OPACITY } from "@/graph/config";

const EDGE_COLOR = new Color(0.4, 0.4, 0.5);
const HIGHLIGHT_COLOR = new Color(0.6, 0.6, 0.8);

const EdgesInner = memo(() => {
  const data = useGraphData((s) => s.data);
  const ready = useSimulation((s) => s.ready);
  const builtRef = useRef(false);

  const baseMaterial = useMemo(
    () =>
      new LineBasicMaterial({
        color: EDGE_COLOR,
        transparent: true,
        opacity: EDGE_OPACITY,
        depthWrite: false,
      }),
    []
  );

  const highlightMaterial = useMemo(
    () =>
      new LineBasicMaterial({
        color: HIGHLIGHT_COLOR,
        transparent: true,
        opacity: 0.15,
        depthWrite: false,
      }),
    []
  );

  const baseGeometry = useMemo(() => new BufferGeometry(), []);
  const highlightGeometry = useMemo(() => new BufferGeometry(), []);

  // Build base edges once
  useFrame(() => {
    if (!ready || builtRef.current || !data?.links || data.links.length === 0)
      return;
    if (nodePositions.size === 0) return;

    const links = data.links;
    const positions = new Float32Array(links.length * 2 * 3);
    let validCount = 0;

    for (let i = 0; i < links.length; i++) {
      const link = links[i];
      const sp = nodePositions.get(link.source);
      const tp = nodePositions.get(link.target);
      if (!sp || !tp) continue;

      const offset = validCount * 6;
      positions[offset] = sp.x;
      positions[offset + 1] = sp.y;
      positions[offset + 2] = sp.z;
      positions[offset + 3] = tp.x;
      positions[offset + 4] = tp.y;
      positions[offset + 5] = tp.z;
      validCount++;
    }

    baseGeometry.setAttribute(
      "position",
      new Float32BufferAttribute(positions.subarray(0, validCount * 6), 3)
    );
    baseGeometry.computeBoundingSphere();
    builtRef.current = true;
  });

  // Update highlight edges reactively
  useFrame(() => {
    const { highlightedFeatureId, highlightedNodeIds } =
      useGraphData.getState();

    if (!highlightedFeatureId || !data?.links) {
      // Clear highlight geometry
      if (highlightGeometry.attributes.position) {
        highlightGeometry.setAttribute(
          "position",
          new Float32BufferAttribute(new Float32Array(0), 3)
        );
      }
      // Restore base opacity
      baseMaterial.opacity = EDGE_OPACITY;
      return;
    }

    // Dim base edges slightly
    baseMaterial.opacity = EDGE_OPACITY * 0.5;

    // Build highlight-only edges
    const links = data.links;
    const positions = new Float32Array(links.length * 2 * 3);
    let count = 0;

    for (const link of links) {
      if (
        !highlightedNodeIds.has(link.source) ||
        !highlightedNodeIds.has(link.target)
      )
        continue;

      const sp = nodePositions.get(link.source);
      const tp = nodePositions.get(link.target);
      if (!sp || !tp) continue;

      const offset = count * 6;
      positions[offset] = sp.x;
      positions[offset + 1] = sp.y;
      positions[offset + 2] = sp.z;
      positions[offset + 3] = tp.x;
      positions[offset + 4] = tp.y;
      positions[offset + 5] = tp.z;
      count++;
    }

    highlightGeometry.setAttribute(
      "position",
      new Float32BufferAttribute(positions.subarray(0, count * 6), 3)
    );
    highlightGeometry.computeBoundingSphere();
  });

  return (
    <>
      <lineSegments geometry={baseGeometry} material={baseMaterial} />
      <lineSegments geometry={highlightGeometry} material={highlightMaterial} />
    </>
  );
});

EdgesInner.displayName = "EdgesInner";

export { EdgesInner as Edges };
