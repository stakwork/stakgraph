import { memo, useMemo, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { Billboard, Text } from "@react-three/drei";
import {
  BufferGeometry,
  Float32BufferAttribute,
  LineBasicMaterial,
  Color,
  Group,
  Vector3,
  MathUtils,
} from "three";
import { useGraphData, type TraceMode } from "@/stores/useGraphData";
import { nodePositions, useSimulation } from "@/stores/useSimulation";
import { EDGE_OPACITY } from "@/graph/config";

const EDGE_COLOR = new Color(0.4, 0.4, 0.5);
const HIGHLIGHT_COLOR = new Color(0.6, 0.6, 0.8);

const TRACE_COLORS: Record<TraceMode, string> = {
  down: "#FFD36B",
  up: "#82D7D0",
  both: "#BCAEFF",
};

const TRACE_LABEL_MAX_COUNT = 28;
const TRACE_LABEL_MIN_EDGE_LENGTH = 120;
const TRACE_LABEL_FADE_START_DISTANCE = 1550;
const TRACE_LABEL_FADE_END_DISTANCE = 2100;
const TRACE_LABEL_OFFSET = 24;
const TRACE_LABEL_MIN_SPACING = 140;

const TRACE_MODE_ICONS: Record<TraceMode, string> = {
  down: "↓",
  up: "↑",
  both: "↕",
};

const EDGE_TYPE_LABELS: Record<string, string> = {
  Calls: "Calls",
  Uses: "Uses",
  Operand: "Operand",
  ArgOf: "Arg Of",
  Contains: "Contains",
  Imports: "Imports",
  Of: "Of",
  Handler: "Handler",
  Includes: "Includes",
  Renders: "Renders",
  ParentOf: "Parent",
  Implements: "Implements",
  NestedIn: "Nested In",
};

function getTraceColor(mode: TraceMode): string {
  return TRACE_COLORS[mode] || TRACE_COLORS.down;
}

function humanizeEdgeType(edgeType?: string): string {
  if (!edgeType) return "Related";
  if (EDGE_TYPE_LABELS[edgeType]) return EDGE_TYPE_LABELS[edgeType];

  const humanized = edgeType
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/_/g, " ")
    .trim();

  return humanized || "Related";
}

const EdgesInner = memo(() => {
  const data = useGraphData((s) => s.data);
  const tracedPath = useGraphData((s) => s.tracedPath);
  const ready = useSimulation((s) => s.ready);
  const builtRef = useRef(false);
  const labelsGroupRef = useRef<Group>(null);
  const cameraRefTarget = useRef(new Vector3());
  const [labelOpacity, setLabelOpacity] = useState(1);

  const baseMaterial = useMemo(
    () =>
      new LineBasicMaterial({
        color: EDGE_COLOR,
        transparent: true,
        opacity: EDGE_OPACITY,
        depthWrite: false,
      }),
    [],
  );

  const highlightMaterial = useMemo(
    () =>
      new LineBasicMaterial({
        color: HIGHLIGHT_COLOR,
        transparent: true,
        opacity: 0.15,
        depthWrite: false,
      }),
    [],
  );

  const traceMaterial = useMemo(
    () =>
      new LineBasicMaterial({
        color: new Color(getTraceColor("down")),
        transparent: true,
        opacity: 0.75,
        depthWrite: false,
        linewidth: 2,
      }),
    [],
  );

  const baseGeometry = useMemo(() => new BufferGeometry(), []);
  const highlightGeometry = useMemo(() => new BufferGeometry(), []);
  const traceGeometry = useMemo(() => new BufferGeometry(), []);

  const traceLabels = useMemo(() => {
    if (!tracedPath || !data?.links || tracedPath.edgeKeys.size === 0)
      return [];

    const labels: Array<{
      key: string;
      text: string;
      position: [number, number, number];
      length: number;
      priority: number;
      textWidth: number;
    }> = [];
    const acceptedPositions: Array<[number, number, number]> = [];

    for (const link of data.links) {
      const edgeKey = `${link.source}->${link.target}`;
      if (!tracedPath.edgeKeys.has(edgeKey)) continue;

      const sp = nodePositions.get(link.source);
      const tp = nodePositions.get(link.target);
      if (!sp || !tp) continue;

      const dx = tp.x - sp.x;
      const dy = tp.y - sp.y;
      const dz = tp.z - sp.z;
      const length = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (length < TRACE_LABEL_MIN_EDGE_LENGTH) continue;

      const labelText = `${TRACE_MODE_ICONS[tracedPath.mode]} ${humanizeEdgeType(link.edge_type)}`;
      const priority =
        link.source === tracedPath.rootId || link.target === tracedPath.rootId
          ? 2
          : 1;

      const midpointX = (sp.x + tp.x) / 2;
      const midpointY = (sp.y + tp.y) / 2;
      const midpointZ = (sp.z + tp.z) / 2;

      const horizontalLength = Math.sqrt(dx * dx + dz * dz) || 1;
      const offsetX = (-dz / horizontalLength) * TRACE_LABEL_OFFSET;
      const offsetZ = (dx / horizontalLength) * TRACE_LABEL_OFFSET;
      const labelPosition: [number, number, number] = [
        midpointX + offsetX,
        midpointY + 10,
        midpointZ + offsetZ,
      ];

      labels.push({
        key: link.ref_id,
        text: labelText,
        position: labelPosition,
        length,
        priority,
        textWidth: Math.max(92, labelText.length * 12),
      });
    }

    labels.sort((a, b) => {
      if (b.priority !== a.priority) return b.priority - a.priority;
      return b.length - a.length;
    });

    const filtered: typeof labels = [];
    for (const label of labels) {
      const overlaps = acceptedPositions.some(([x, y, z]) => {
        const dx = label.position[0] - x;
        const dy = label.position[1] - y;
        const dz = label.position[2] - z;
        return Math.sqrt(dx * dx + dy * dy + dz * dz) < TRACE_LABEL_MIN_SPACING;
      });

      if (overlaps) continue;

      filtered.push(label);
      acceptedPositions.push(label.position);

      if (filtered.length >= TRACE_LABEL_MAX_COUNT) break;
    }

    return filtered;
  }, [data?.links, tracedPath]);

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
      new Float32BufferAttribute(positions.subarray(0, validCount * 6), 3),
    );
    baseGeometry.computeBoundingSphere();
    builtRef.current = true;
  });

  // Update highlight edges reactively
  useFrame(() => {
    const { highlightedFeatureId, highlightedNodeIds } =
      useGraphData.getState();

    const { tracedPath } = useGraphData.getState();
    const hasTrace = tracedPath && tracedPath.edgeKeys.size > 0;

    if (!highlightedFeatureId || !data?.links) {
      // Clear highlight geometry
      if (highlightGeometry.attributes.position) {
        highlightGeometry.setAttribute(
          "position",
          new Float32BufferAttribute(new Float32Array(0), 3),
        );
      }
      // Restore base opacity unless trace is dimming it
      baseMaterial.opacity = hasTrace ? EDGE_OPACITY * 0.3 : EDGE_OPACITY;
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
      new Float32BufferAttribute(positions.subarray(0, count * 6), 3),
    );
    highlightGeometry.computeBoundingSphere();
  });

  // Build trace path edges
  useFrame(({ camera }) => {
    const tracedPathState = useGraphData.getState().tracedPath;

    if (labelsGroupRef.current) {
      if (!tracedPathState || traceLabels.length === 0) {
        labelsGroupRef.current.visible = false;
        if (labelOpacity !== 0) setLabelOpacity(0);
      } else {
        const rootPos = nodePositions.get(tracedPathState.rootId);
        if (rootPos) {
          cameraRefTarget.current.set(rootPos.x, rootPos.y, rootPos.z);
          const distance = camera.position.distanceTo(cameraRefTarget.current);
          const nextOpacity =
            1 -
            MathUtils.clamp(
              (distance - TRACE_LABEL_FADE_START_DISTANCE) /
                (TRACE_LABEL_FADE_END_DISTANCE -
                  TRACE_LABEL_FADE_START_DISTANCE),
              0,
              1,
            );
          labelsGroupRef.current.visible = nextOpacity > 0.02;
          if (Math.abs(nextOpacity - labelOpacity) > 0.03) {
            setLabelOpacity(nextOpacity);
          }
        } else {
          labelsGroupRef.current.visible = true;
          if (labelOpacity !== 1) setLabelOpacity(1);
        }
      }
    }

    const tracedPath = tracedPathState;
    if (!tracedPath || tracedPath.edgeKeys.size === 0 || !data?.links) {
      if (traceGeometry.attributes.position) {
        traceGeometry.setAttribute(
          "position",
          new Float32BufferAttribute(new Float32Array(0), 3),
        );
      }
      return;
    }

    traceMaterial.color.set(getTraceColor(tracedPath.mode));

    const links = data.links;
    const positions = new Float32Array(links.length * 2 * 3);
    let count = 0;

    for (const link of links) {
      const edgeKey = `${link.source}->${link.target}`;
      if (!tracedPath.edgeKeys.has(edgeKey)) continue;

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

    traceGeometry.setAttribute(
      "position",
      new Float32BufferAttribute(positions.subarray(0, count * 6), 3),
    );
    traceGeometry.computeBoundingSphere();
  });

  return (
    <>
      <lineSegments geometry={baseGeometry} material={baseMaterial} />
      <lineSegments geometry={highlightGeometry} material={highlightMaterial} />
      <lineSegments geometry={traceGeometry} material={traceMaterial} />
      {tracedPath && traceLabels.length > 0 && (
        <group ref={labelsGroupRef}>
          {traceLabels.map((label) => (
            <Billboard key={label.key} position={label.position}>
              <group>
                <mesh position={[0, 0, -1]}>
                  <planeGeometry args={[label.textWidth, 34]} />
                  <meshBasicMaterial
                    color="#0B1020"
                    transparent
                    opacity={0.55 * labelOpacity}
                    depthWrite={false}
                  />
                </mesh>
                <Text
                  fontSize={20}
                  color={getTraceColor(tracedPath.mode)}
                  anchorX="center"
                  anchorY="middle"
                  material-toneMapped={false}
                  fillOpacity={labelOpacity}
                  outlineWidth={0.025}
                  outlineColor="#020617"
                >
                  {label.text}
                </Text>
              </group>
            </Billboard>
          ))}
        </group>
      )}
    </>
  );
});

EdgesInner.displayName = "EdgesInner";

export { EdgesInner as Edges };
