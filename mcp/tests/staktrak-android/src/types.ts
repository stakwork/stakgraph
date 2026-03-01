export type AndroidSelector = {
  resourceId?: string;
  accessibilityId?: string;
  text?: string;
  xpath?: string;
};

export type StartSessionInput = {
  packageName: string;
  activity?: string;
  deviceName: string;
};

export type AppiumSessionMeta = {
  packageName: string;
  activity?: string;
  deviceName: string;
};

export type RecordedAction =
  | {
      type: "tap";
      timestamp: number;
      selector?: AndroidSelector;
      x?: number;
      y?: number;
    }
  | {
      type: "type";
      timestamp: number;
      selector: AndroidSelector;
      text: string;
      replace: boolean;
    }
  | {
      type: "swipe";
      timestamp: number;
      startX: number;
      startY: number;
      endX: number;
      endY: number;
      durationMs: number;
    }
  | {
      type: "back";
      timestamp: number;
    }
  | {
      type: "home";
      timestamp: number;
    };

export type RecordedActionInput =
  | {
      type: "tap";
      selector?: AndroidSelector;
      x?: number;
      y?: number;
    }
  | {
      type: "type";
      selector: AndroidSelector;
      text: string;
      replace: boolean;
    }
  | {
      type: "swipe";
      startX: number;
      startY: number;
      endX: number;
      endY: number;
      durationMs: number;
    }
  | {
      type: "back";
    }
  | {
      type: "home";
    };

export type RecordingContext = {
  packageName: string;
  activity?: string;
  deviceName: string;
};

export type ReplayEvent =
  | { type: "started"; total: number }
  | {
      type: "progress";
      current: number;
      total: number;
      action: RecordedAction;
      screenshot: string;
    }
  | {
      type: "error";
      current: number;
      total: number;
      action: RecordedAction;
      error: string;
      screenshot?: string;
    }
  | { type: "completed"; total: number; errors: number };

export type ReplaySummary = {
  total: number;
  completed: number;
  errors: Array<{ index: number; message: string }>;
};