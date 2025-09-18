import {
  ReplayStatus,
  PlaywrightAction,
  PlaywrightReplayState,
} from "../types";
import { parsePlaywrightTest } from "./parser";
import { executePlaywrightAction, getActionDescription } from "./executor";

let playwrightReplayRef = {
  current: null as {
    actions: PlaywrightAction[];
    status: ReplayStatus;
    currentActionIndex: number;
    testCode: string;
    errors: string[];
    timeouts: number[];
  } | null,
};


export function startPlaywrightReplay(testCode: string): void {
  try {
    const actions = parsePlaywrightTest(testCode);

    if (actions.length === 0) {
      throw new Error("No valid actions found in test code");
    }

    try {
      if (typeof (window as any).__stakTrakInitAnnotations === 'function') {
        setTimeout(() => {
          try {
            (window as any).__stakTrakInitAnnotations();
            console.log('[staktrak-replay] Annotations initialized for replay');
          } catch (e) {
            console.warn('[staktrak-replay] Failed to initialize annotations:', e);
          }
        }, 100);
      }
    } catch (e) {
      console.warn('[staktrak-replay] Failed to initialize annotations:', e);
    }

    playwrightReplayRef.current = {
      actions,
      status: ReplayStatus.PLAYING,
      currentActionIndex: 0,
      testCode,
      errors: [],
      timeouts: [],
    };

    window.parent.postMessage(
      {
        type: "staktrak-playwright-replay-started",
        totalActions: actions.length,
        actions: actions,
      },
      "*"
    );

    setTimeout(() => {
      executeNextPlaywrightAction();
    }, 200);
  } catch (error) {
    window.parent.postMessage(
      {
        type: "staktrak-playwright-replay-error",
        error: error instanceof Error ? error.message : "Unknown error",
      },
      "*"
    );
  }
}

async function executeNextPlaywrightAction(): Promise<void> {
  const state = playwrightReplayRef.current;
  if (!state || state.status !== ReplayStatus.PLAYING) {
    return;
  }

  if (state.currentActionIndex >= state.actions.length) {
    state.status = ReplayStatus.COMPLETED;
    window.parent.postMessage(
      {
        type: "staktrak-playwright-replay-completed",
      },
      "*"
    );
    return;
  }

  const action = state.actions[state.currentActionIndex];

  try {
    window.parent.postMessage(
      {
        type: "staktrak-playwright-replay-progress",
        current: state.currentActionIndex + 1,
        total: state.actions.length,
        currentAction: {
          ...action,
          description: getActionDescription(action),
        },
      },
      "*"
    );

    await executePlaywrightAction(action);

    state.currentActionIndex++;

    setTimeout(() => {
      executeNextPlaywrightAction();
    }, 300);
  } catch (error) {
    state.errors.push(
      `Action ${state.currentActionIndex + 1}: ${error instanceof Error ? error.message : "Unknown error"}`
    );

    state.currentActionIndex++;

    window.parent.postMessage(
      {
        type: "staktrak-playwright-replay-error",
        error: error instanceof Error ? error.message : "Unknown error",
        actionIndex: state.currentActionIndex - 1,
        action: action,
      },
      "*"
    );

    executeNextPlaywrightAction();
  }
}

export function pausePlaywrightReplay(): void {
  const state = playwrightReplayRef.current;
  if (state) {
    state.status = ReplayStatus.PAUSED;

    state.timeouts.forEach((id) => clearTimeout(id as any));
    state.timeouts = [];

    window.parent.postMessage(
      { type: "staktrak-playwright-replay-paused" },
      "*"
    );
  }
}

export function resumePlaywrightReplay(): void {
  const state = playwrightReplayRef.current;
  if (state && state.status === ReplayStatus.PAUSED) {
    state.status = ReplayStatus.PLAYING;

    executeNextPlaywrightAction();

    window.parent.postMessage(
      { type: "staktrak-playwright-replay-resumed" },
      "*"
    );
  }
}

export function stopPlaywrightReplay(): void {
  const state = playwrightReplayRef.current;
  if (state) {
    state.status = ReplayStatus.IDLE;

    state.timeouts.forEach((id) => clearTimeout(id as any));
    state.timeouts = [];

    window.parent.postMessage(
      { type: "staktrak-playwright-replay-stopped" },
      "*"
    );
  }
}

export function getPlaywrightReplayState(): PlaywrightReplayState | null {
  const state = playwrightReplayRef.current;
  if (!state) return null;

  return {
    actions: state.actions,
    status: state.status,
    currentActionIndex: state.currentActionIndex,
    testCode: state.testCode,
    errors: state.errors,
  };
}

export function initPlaywrightReplay(): void {
  // Lightweight SPA history instrumentation (idempotent)
  try {
    if (!(window as any).__stakTrakHistoryInstrumented) {
      const fire = () => {
        try {
          const detail = { href: window.location.href, path: window.location.pathname, ts: Date.now() };
          const ev = new CustomEvent('staktrak-history-change', { detail });
          window.dispatchEvent(ev);
        } catch {}
      };
      const origPush = history.pushState;
      const origReplace = history.replaceState;
      history.pushState = function(this: any, ...args: any[]) {
        const ret = origPush.apply(this, args as any);
        setTimeout(fire, 0);
        return ret;
      } as any;
      history.replaceState = function(this: any, ...args: any[]) {
        const ret = origReplace.apply(this, args as any);
        setTimeout(fire, 0);
        return ret;
      } as any;
      window.addEventListener('popstate', fire, { passive: true });
      // Initial fire to mark baseline
      setTimeout(fire, 0);
      (window as any).__stakTrakHistoryInstrumented = true;
    }
  } catch {}

  window.addEventListener("message", (event) => {
    const { data } = event;

    if (!data || !data.type) return;

    switch (data.type) {
      case "staktrak-playwright-replay-start":
        if (data.testCode) {
          startPlaywrightReplay(data.testCode);
        }
        break;

      case "staktrak-playwright-replay-pause":
        pausePlaywrightReplay();
        break;

      case "staktrak-playwright-replay-resume":
        resumePlaywrightReplay();
        break;

      case "staktrak-playwright-replay-stop":
        stopPlaywrightReplay();
        break;

      case "staktrak-playwright-replay-ping":
        const currentState = getPlaywrightReplayState();
        window.parent.postMessage(
          {
            type: "staktrak-playwright-replay-pong",
            state: currentState,
          },
          "*"
        );
        break;
    }
  });
}
