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
  pageNavigation: boolean;
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
    clickDetails: Array<[number, number, string, number]>;
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
  postMessageListeners: Array<() => void>;
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

export interface ReplayOptions {
  speed?: number;
  initialDelay?: number;
  minDelay?: number;
  maxDelay?: number;
}
