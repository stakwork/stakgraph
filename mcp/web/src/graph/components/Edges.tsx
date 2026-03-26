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

const EDGE_COLOR = new Color(0.4, 0.4, 0.5);

const EdgesInner = memo(() => {
  const data = useGraphData((s) => s.data);
  const ready = useSimulation((s) => s.ready);
  const geoRef = useRef<BufferGeometry | null>(null);
  const builtRef = useRef(false);

  const material = useMemo(
    () =>
      new LineBasicMaterial({
        color: EDGE_COLOR,
        transparent: true,
        opacity: 0.06,
        depthWrite: false,
      }),
    []
  );

  const geometry = useMemo(() => new BufferGeometry(), []);

  useFrame(() => {
    if (!ready || builtRef.current || !data?.links || data.links.length === 0)
      return;
    if (nodePositions.size === 0) return;

    const links = data.links;
    // 2 points per line segment, 3 floats per point
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

    geometry.setAttribute(
      "position",
      new Float32BufferAttribute(positions.subarray(0, validCount * 6), 3)
    );
    geometry.computeBoundingSphere();
    geoRef.current = geometry;
    builtRef.current = true;
  });

  return <lineSegments geometry={geometry} material={material} />;
});

EdgesInner.displayName = "EdgesInner";

export { EdgesInner as Edges };
