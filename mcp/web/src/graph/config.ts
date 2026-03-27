// Layer order (top to bottom) — only these node types are rendered
export const LAYER_ORDER = [
  "Repository",
  "Package",
  "Language",
  "Feature",
  "Directory",
  "Library",
  "File",
  "Import",
  "PullRequest",
  "Commit",
  "Trait",
  "Class",
  "Instance",
  "Function",
  "Datamodel",
  "Endpoint",
  "Request",
  "Var",
  "Page",
  "UnitTest",
  "IntegrationTest",
  "E2eTest",
];

// Camera
export const INITIAL_CAMERA_POSITION: [number, number, number] = [2000, 1000, 6000];
export const CAMERA_NEAR = 1;
export const CAMERA_FAR = 30000;
export const CAMERA_MIN_DISTANCE = 50;
export const CAMERA_MAX_DISTANCE = 15000;
export const CAMERA_SMOOTH_TIME = 0.8;
export const AUTO_ROTATE_SPEED = 0.5; // degrees per second

// Layout
export const LAYER_SPACING = 700;
export const GRID_SPACING = 300;
export const GRID_PADDING = 100;

// Edges
export const EDGE_OPACITY = 0.04;

// Node size
export const NODE_SIZE = 20;
