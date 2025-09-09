export interface Config {
  userInfo: boolean;
  clicks: boolean;
  mouseMovement: boolean;
  mouseMovementInterval: number;
  mouseScroll: boolean;
  timeCount: boolean;
  clearAfterProcess: boolean;
  windowResize: boolean;
  visibilitychange: boolean;
  keyboardActivity: boolean;
  formInteractions: boolean;
  touchEvents: boolean;
  audioVideoInteraction: boolean;
  customEventRegistration: boolean;
  inputDebounceDelay: number;
  multiClickInterval: number;
  filterAssertionClicks: boolean;
  processData: (results: Results) => void;
}

export interface Assertion {
  type: string;
  selector: string;
  value: string;
  timestamp: number;
}

export interface ComponentInfo {
  name: string;
  level: number;
  type: 'function' | 'class';
}

// Add to types.ts
export interface ClickDetail {
  x: number;
  y: number;
  timestamp: number;
  selectors: {
    primary: string; // Main CSS selector
    fallbacks: string[]; // Alternative selectors
    text?: string; // Button/link text content
    ariaLabel?: string; // aria-label if present
    title?: string; // title attribute
    role?: string; // role attribute
    tagName: string; // Element tag name
    xpath?: string; // XPath as last resort
  };
  elementInfo: {
    tagName: string;
    id?: string;
    className?: string;
    attributes: Record<string, string>;
  };
}

export interface Results {
  userInfo?: {
    url: string;
    userAgent: string;
    platform: string;
    windowSize: [number, number];
  };
  time?: {
    startedAt: number;
    completedAt: number;
    totalSeconds: number;
  };
  pageNavigation: Array<{ type: string; url: string; timestamp: number }>;
  clicks: {
    clickCount: number;
    clickDetails: ClickDetail[];
  };
  keyboardActivities: Array<[string, number]>;
  mouseMovement: Array<[number, number, number]>;
  mouseScroll: Array<[number, number, number]>;
  inputChanges: Array<{
    elementSelector: string;
    value: string;
    timestamp: number;
    action: string;
  }>;
  focusChanges: Array<{
    elementSelector: string;
    type: string;
    timestamp: number;
  }>;
  visibilitychanges: Array<[string, number]>;
  windowSizes: Array<[number, number, number]>;
  formElementChanges: Array<{
    elementSelector: string;
    type: string;
    checked?: boolean;
    value: string;
    text?: string;
    timestamp: number;
  }>;
  touchEvents: Array<{
    type: string;
    x: number;
    y: number;
    timestamp: number;
  }>;
  audioVideoInteractions: any[];
  assertions: Assertion[];
}

export interface Memory {
  mousePosition: [number, number, number];
  inputDebounceTimers: Record<string, NodeJS.Timeout>;
  selectionMode: boolean;
  assertionDebounceTimer: NodeJS.Timeout | null;
  assertions: Assertion[];
  mutationObserver: MutationObserver | null;
  mouseInterval: NodeJS.Timeout | null;
  listeners: Array<() => void>;
  alwaysListeners: Array<() => void>;
}

export enum ActionType {
  CLICK = "click",
  INPUT = "input",
  SELECT = "select",
  CHECK = "check",
  UNCHECK = "uncheck",
  WAIT = "wait",
}

export enum ReplayStatus {
  IDLE = "idle",
  PLAYING = "playing",
  PAUSED = "paused",
  COMPLETED = "completed",
}

export interface ReplayAction {
  type: ActionType;
  selector: string;
  timestamp: number;
  x?: number;
  y?: number;
  value?: string;
}

export interface ReplayState {
  actions: ReplayAction[];
  status: ReplayStatus;
  currentActionIndex: number;
  speed: number;
  overlayVisible: boolean;
}

export interface ReplayProgress {
  current: number;
  total: number;
}

export interface PlaywrightAction {
  type: 'goto' | 'click' | 'fill' | 'check' | 'uncheck' | 'selectOption' | 'waitForTimeout' | 'expect' | 'setViewportSize' | 'waitForLoadState' | 'waitForSelector' | 'waitFor' | 'hover' | 'focus' | 'blur' | 'scrollIntoView';
  selector?: string;
  value?: string | number;
  options?: Record<string, any>;
  expectation?: 'toBeVisible' | 'toContainText' | 'toBeChecked' | 'not.toBeChecked' | 'toHaveText' | 'toHaveCount';
  comment?: string;
  lineNumber?: number;
}

export interface PlaywrightReplayState {
  actions: PlaywrightAction[];
  status: ReplayStatus;
  currentActionIndex: number;
  testCode: string;
  errors: string[];
}