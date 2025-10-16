import {
  ReplayStatus,
  PlaywrightAction,
  PlaywrightReplayState,
} from "../types";
import { parsePlaywrightTest } from "./parser";
import { executePlaywrightAction, getActionDescription } from "./executor";
import { domToDataUrl } from 'modern-screenshot';

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

/**
 * Capture screenshot and send to parent window
 */
async function captureScreenshot(actionIndex: number, url: string): Promise<void> {
  try {
    const dataUrl = await domToDataUrl(document.body, {
      quality: 0.8,
      type: 'image/jpeg',
      scale: 1,
      backgroundColor: '#ffffff'
    });

    const timestamp = Date.now();
    const id = `${timestamp}-${actionIndex}`;

    // Send screenshot directly to parent window
    window.parent.postMessage(
      {
        type: 'staktrak-playwright-screenshot-captured',
        screenshot: dataUrl,
        actionIndex,
        url,
        timestamp,
        id
      },
      '*'
    );
  } catch (error) {
    console.error(`[Screenshot] Error capturing for actionIndex=${actionIndex}:`, error);
  }
}


export async function startPlaywrightReplay(testCode: string): Promise<void> {
  try {
    const actions = parsePlaywrightTest(testCode);

    if (actions.length === 0) {
      throw new Error("No valid actions found in test code");
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

    executeNextPlaywrightAction();
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

    // Capture screenshot after waitForURL actions (meaningful page changes after interactions)
    if (action.type === "waitForURL") {
      await captureScreenshot(state.currentActionIndex, window.location.href);
    }

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
