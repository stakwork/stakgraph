// Graph node as returned by the stakgraph API
export interface GraphNode {
  node_type: string;
  ref_id: string;
  properties: {
    name: string;
    file?: string;
    body?: string;
    start?: number;
    end?: number;
    docs?: string;
    [key: string]: unknown;
  };
  date_added_to_graph?: string | number;
}

// Extended node used in the simulation/rendering
export interface NodeExtended extends GraphNode {
  x: number;
  y: number;
  z: number;
  fx?: number | null;
  fy?: number | null;
  fz?: number | null;
  vx?: number;
  vy?: number;
  vz?: number;
  sources?: string[];
  targets?: string[];
  index?: number;
  [key: string]: unknown;
}

// Graph edge as returned by the stakgraph API
export interface GraphEdge {
  edge_type: string;
  source: string;
  target: string;
  ref_id?: string;
  properties?: Record<string, unknown>;
}

// Link used in rendering (source/target resolved to ref_ids)
export interface Link {
  source: string;
  target: string;
  ref_id: string;
  edge_type: string;
}

export interface GraphData {
  nodes: NodeExtended[];
  links: Link[];
}

export interface LinkPosition {
  sx: number;
  sy: number;
  sz: number;
  tx: number;
  ty: number;
  tz: number;
}

// API response shapes
export interface GraphApiResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  status: string;
  meta?: {
    node_types: string[];
    counts: Record<string, number>;
  };
}
