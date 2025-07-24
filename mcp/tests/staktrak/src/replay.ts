import {
  ActionType,
  ReplayStatus,
  ReplayAction,
  ReplayOptions,
  ReplayState,
} from "./types";

const DEFAULT_SPEED = 1;
const MIN_DELAY = 1000;
const MAX_DELAY = 5000;
const INITIAL_DELAY = 2000;

interface TimeoutRef {
  current: NodeJS.Timeout | null;
}

interface ElementRef {
  current: HTMLElement | null;
}

interface ActionsRef {
  current: ReplayAction[];
}

interface StatusRef {
  current: ReplayStatus;
}

interface SpeedRef {
  current: number;
}

interface AnimationRef {
  current: number | null;
}

interface IsTypingRef {
  current: boolean;
}

interface AllTimeoutsRef {
  current: NodeJS.Timeout[];
}

let cursorRef: ElementRef = { current: null };
let statusRef: StatusRef = { current: ReplayStatus.IDLE };
let speedRef: SpeedRef = { current: DEFAULT_SPEED };
let actionsRef: ActionsRef = { current: [] };
let currentActionIndexRef = { current: 0 };
let isTypingRef: IsTypingRef = { current: false };
let timeoutIdsRef: AllTimeoutsRef = { current: [] };
let timeoutRef: TimeoutRef = { current: null };

function registerTimeout(timeoutId: NodeJS.Timeout): NodeJS.Timeout {
  timeoutIdsRef.current.push(timeoutId);
  return timeoutId;
}

function clearAllTimeouts(): void {
  timeoutIdsRef.current.forEach(clearTimeout);
  timeoutIdsRef.current = [];
}

export function convertToReplayActions(trackingData: any): ReplayAction[] {
  if (!trackingData) {
    console.error("No tracking data provided to convertToReplayActions");
    return [];
  }
  console.log("Converting tracking data to replay actions:", trackingData);

  const actions: ReplayAction[] = [];

  try {
    const { clicks, inputChanges, formElementChanges } = trackingData;

    if (clicks?.clickDetails?.length) {
      clicks.clickDetails.forEach(
        ([x, y, selector, timestamp]: [number, number, string, number]) => {
          if (!selector || selector === "undefined" || selector === "null") {
            console.warn("Skipping click with invalid selector:", selector);
            return;
          }

          let cleanSelector = selector.trim();

          try {
            document.querySelector(cleanSelector);

            actions.push({
              type: ActionType.CLICK,
              selector: cleanSelector,
              timestamp,
              x,
              y,
            });
          } catch (e) {
            console.warn(
              `Invalid selector in click event: ${cleanSelector}. Attempting to fix.`
            );

            if (cleanSelector.includes("data-testid=")) {
              try {
                const testIdMatch = cleanSelector.match(
                  /data-testid="([^"]+)"/
                );
                if (testIdMatch && testIdMatch[1]) {
                  const simpleSelector = `[data-testid="${testIdMatch[1]}"]`;
                  try {
                    document.querySelector(simpleSelector);
                    actions.push({
                      type: ActionType.CLICK,
                      selector: simpleSelector,
                      timestamp,
                      x,
                      y,
                    });
                  } catch (err) {
                    actions.push({
                      type: ActionType.CLICK,
                      selector: "[data-testid]",
                      timestamp,
                      x,
                      y,
                    });
                  }
                }
              } catch (err) {
                console.error(
                  "Failed to create valid selector from data-testid",
                  err
                );
              }
            } else if (cleanSelector.includes("class=")) {
              try {
                const classMatch = cleanSelector.match(/class="([^"]+)"/);
                if (classMatch && classMatch[1]) {
                  const classNames = classMatch[1].split(" ");
                  if (classNames.length > 0) {
                    const simpleSelector = `.${classNames[0]}`;
                    document.querySelector(simpleSelector);
                    actions.push({
                      type: ActionType.CLICK,
                      selector: simpleSelector,
                      timestamp,
                      x,
                      y,
                    });
                  }
                }
              } catch (err) {
                console.error(
                  "Failed to create valid selector from class",
                  err
                );
              }
            } else if (cleanSelector.includes("id=")) {
              try {
                const idMatch = cleanSelector.match(/id="([^"]+)"/);
                if (idMatch && idMatch[1]) {
                  const simpleSelector = `#${idMatch[1]}`;
                  document.querySelector(simpleSelector);
                  actions.push({
                    type: ActionType.CLICK,
                    selector: simpleSelector,
                    timestamp,
                    x,
                    y,
                  });
                }
              } catch (err) {
                console.error("Failed to create valid selector from id", err);
              }
            } else {
              const tagMatch = cleanSelector.match(/^([a-zA-Z]+)/);
              if (tagMatch && tagMatch[1]) {
                try {
                  const simpleSelector = tagMatch[1];
                  document.querySelector(simpleSelector);
                  actions.push({
                    type: ActionType.CLICK,
                    selector: simpleSelector,
                    timestamp,
                    x,
                    y,
                  });
                } catch (err) {
                  console.error(
                    "Failed to create valid selector from tag name",
                    err
                  );
                }
              }
            }
          }
        }
      );
    }

    if (inputChanges?.length) {
      const completedInputs = inputChanges.filter(
        (change: any) => change.action === "complete" || !change.action
      );

      completedInputs.forEach((change: any) => {
        if (
          !change.elementSelector.includes('type="checkbox"') &&
          !change.elementSelector.includes('type="radio"')
        ) {
          try {
            document.querySelector(change.elementSelector);
            actions.push({
              type: ActionType.INPUT,
              selector: change.elementSelector,
              value: change.value,
              timestamp: change.timestamp,
            });
          } catch (e) {
            console.warn(
              `Invalid selector in input: ${change.elementSelector}, attempting to fix`
            );

            if (change.elementSelector.includes("data-testid=")) {
              const testIdMatch = change.elementSelector.match(
                /data-testid="([^"]+)"/
              );
              if (testIdMatch && testIdMatch[1]) {
                actions.push({
                  type: ActionType.INPUT,
                  selector: `[data-testid="${testIdMatch[1]}"]`,
                  value: change.value,
                  timestamp: change.timestamp,
                });
              }
            }
          }
        }
      });
    }

    if (formElementChanges?.length) {
      formElementChanges.forEach((change: any) => {
        if (!change.elementSelector) return;

        try {
          document.querySelector(change.elementSelector);

          if (change.type === "checkbox" || change.type === "radio") {
            actions.push({
              type: change.checked ? ActionType.CHECK : ActionType.UNCHECK,
              selector: change.elementSelector,
              value: change.value,
              timestamp: change.timestamp,
            });
          } else if (change.type === "select") {
            actions.push({
              type: ActionType.SELECT,
              selector: change.elementSelector,
              value: change.value,
              timestamp: change.timestamp,
            });
          }
        } catch (e) {
          console.warn(
            `Invalid selector in form element: ${change.elementSelector}, attempting to fix`
          );

          if (change.elementSelector.includes("data-testid=")) {
            const testIdMatch = change.elementSelector.match(
              /data-testid="([^"]+)"/
            );
            if (testIdMatch && testIdMatch[1]) {
              const selector = `[data-testid="${testIdMatch[1]}"]`;
              if (change.type === "checkbox" || change.type === "radio") {
                actions.push({
                  type: change.checked ? ActionType.CHECK : ActionType.UNCHECK,
                  selector,
                  value: change.value,
                  timestamp: change.timestamp,
                });
              } else if (change.type === "select") {
                actions.push({
                  type: ActionType.SELECT,
                  selector,
                  value: change.value,
                  timestamp: change.timestamp,
                });
              }
            }
          }
        }
      });
    }
  } catch (e) {
    console.error("Error processing tracking data", e);
  }

  if (actions.length === 0) {
    console.warn("No actions extracted from tracking data");
  }

  actions.sort((a, b) => a.timestamp - b.timestamp);

  for (let i = 1; i < actions.length; i++) {
    if (actions[i].timestamp - actions[i - 1].timestamp < 600) {
      actions[i].timestamp = actions[i - 1].timestamp + 600;
    }
  }

  console.log("Converted replay actions:", actions);
  return actions;
}

export function findElement(selector: string): Element | null {
  let element = document.querySelector(selector);

  if (!element) {
    if (selector.includes("data-testid=")) {
      const testId = selector.match(/data-testid="([^"]+)"/)?.[1];
      if (testId) {
        element = document.querySelector(`[data-testid="${testId}"]`);
      }
    }

    if (!element && selector.includes(".")) {
      const classes = selector.match(/\.([^\s.#\[\]]+)/g);
      if (classes && classes.length > 0) {
        const className = classes[0].substring(1);
        element = document.querySelector(`.${className}`);
      }
    }

    if (!element && selector.includes("#")) {
      const ids = selector.match(/#([^\s.#\[\]]+)/g);
      if (ids && ids.length > 0) {
        const id = ids[0].substring(1);
        element = document.querySelector(`#${id}`);
      }
    }
  }

  return element;
}

export function createCursor(): HTMLElement {
  const cursor = document.createElement("div");
  cursor.className = "replay-cursor";
  cursor.style.position = "fixed";
  cursor.style.width = "24px";
  cursor.style.height = "24px";
  cursor.style.borderRadius = "50%";
  cursor.style.backgroundColor = "rgba(255, 0, 0, 0.6)";
  cursor.style.border = "2px solid white";
  cursor.style.boxShadow = "0 0 10px rgba(255, 0, 0, 0.8)";
  cursor.style.zIndex = "9999";
  cursor.style.pointerEvents = "none";
  cursor.style.display = "none";
  cursor.style.transform = "translate(-50%, -50%)";
  cursor.style.transition =
    "transform 0.5s ease-in-out, left 0.5s ease-in-out, top 0.5s ease-in-out";
  document.body.appendChild(cursor);
  return cursor;
}

export function createReplayStyles(): HTMLStyleElement {
  const style = document.createElement("style");
  style.textContent = `
    @keyframes click-ripple {
      0% {
        transform: translate(-50%, -50%) scale(1);
        opacity: 1;
      }
      100% {
        transform: translate(-50%, -50%) scale(8);
        opacity: 0;
      }
    }
    
    @keyframes pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.03); }
      100% { transform: scale(1); }
    }
    
    .replay-pulse {
      animation: pulse 0.5s ease-in-out infinite;
    }
  `;
  document.head.appendChild(style);
  return style;
}

export function showClickEffect(cursorRef: ElementRef): void {
  if (!cursorRef.current) return;

  const ripple = document.createElement("div");
  ripple.className = "click-ripple";
  ripple.style.position = "fixed";
  ripple.style.left = cursorRef.current.style.left;
  ripple.style.top = cursorRef.current.style.top;
  ripple.style.transform = "translate(-50%, -50%)";
  ripple.style.width = "20px";
  ripple.style.height = "20px";
  ripple.style.background = "rgba(255, 0, 0, 0.5)";
  ripple.style.borderRadius = "50%";
  ripple.style.zIndex = "9998";
  ripple.style.pointerEvents = "none";
  ripple.style.animation = "click-ripple 1s ease-out forwards";
  document.body.appendChild(ripple);

  cursorRef.current.style.transform = "translate(-50%, -50%) scale(0.8)";
  setTimeout(() => {
    if (cursorRef.current) {
      cursorRef.current.style.transform = "translate(-50%, -50%) scale(1)";
    }
  }, 200);

  setTimeout(() => {
    if (ripple.parentNode) {
      ripple.parentNode.removeChild(ripple);
    }
  }, 1000);
}

export function highlightElement(element: Element, speedRef: SpeedRef): void {
  const originalOutline = (element as HTMLElement).style.outline;
  const originalBoxShadow = (element as HTMLElement).style.boxShadow;
  const originalZIndex = (element as HTMLElement).style.zIndex;
  const originalTransition = (element as HTMLElement).style.transition;

  (element as HTMLElement).style.transition = "all 0.3s ease-in-out";
  (element as HTMLElement).style.outline = "3px solid #ff3333";
  (element as HTMLElement).style.boxShadow = "0 0 15px rgba(255, 51, 51, 0.7)";
  (element as HTMLElement).style.zIndex = "1000";

  element.classList.add("replay-pulse");

  setTimeout(() => {
    if (element) {
      (element as HTMLElement).style.outline = originalOutline;
      (element as HTMLElement).style.boxShadow = originalBoxShadow;
      (element as HTMLElement).style.zIndex = originalZIndex;
      (element as HTMLElement).style.transition = originalTransition;
      element.classList.remove("replay-pulse");
    }
  }, 2000 / speedRef.current);
}

export function moveCursorToElement(
  element: Element | null,
  cursorRef: ElementRef,
  statusRef: StatusRef
): Promise<void> {
  return new Promise((resolve) => {
    if (
      !element ||
      !cursorRef.current ||
      statusRef.current !== ReplayStatus.PLAYING
    ) {
      resolve();
      return;
    }

    const rect = element.getBoundingClientRect();
    const targetX = rect.left + rect.width / 2;
    const targetY = rect.top + rect.height / 2;

    cursorRef.current.style.display = "block";

    element.scrollIntoView({ behavior: "smooth", block: "center" });

    setTimeout(() => {
      if (cursorRef.current) {
        cursorRef.current.style.left = `${targetX}px`;
        cursorRef.current.style.top = `${targetY}px`;
      }

      setTimeout(resolve, 800);
    }, 400);
  });
}

export function typeText(
  element: HTMLInputElement,
  value: string,
  speedRef: SpeedRef,
  statusRef: StatusRef,
  isTypingRef: IsTypingRef,
  registerTimeout: (timeout: NodeJS.Timeout) => NodeJS.Timeout
): Promise<void> {
  return new Promise((resolve) => {
    if (statusRef.current !== ReplayStatus.PLAYING) {
      resolve();
      return;
    }

    isTypingRef.current = true;
    element.focus();
    element.value = "";

    let index = 0;
    const typeChar = () => {
      if (statusRef.current !== ReplayStatus.PLAYING) {
        isTypingRef.current = false;
        resolve();
        return;
      }

      if (index < value.length) {
        element.value += value[index];
        element.dispatchEvent(new Event("input", { bubbles: true }));
        index++;
        registerTimeout(setTimeout(typeChar, 100 / speedRef.current));
      } else {
        element.dispatchEvent(new Event("change", { bubbles: true }));
        isTypingRef.current = false;
        resolve();
      }
    };

    typeChar();
  });
}

export function selectOption(
  element: HTMLSelectElement,
  value: string,
  speedRef: SpeedRef,
  statusRef: StatusRef,
  registerTimeout: (timeout: NodeJS.Timeout) => NodeJS.Timeout
): Promise<void> {
  return new Promise((resolve) => {
    if (statusRef.current !== ReplayStatus.PLAYING) {
      resolve();
      return;
    }

    element.focus();

    registerTimeout(
      setTimeout(() => {
        if (statusRef.current === ReplayStatus.PLAYING) {
          element.value = value;
          element.dispatchEvent(new Event("change", { bubbles: true }));
        }
        resolve();
      }, 500 / speedRef.current)
    );
  });
}

export async function executeAction(
  index: number,
  actionsRef: ActionsRef,
  statusRef: StatusRef,
  cursorRef: ElementRef,
  speedRef: SpeedRef,
  isTypingRef: IsTypingRef,
  registerTimeout: (timeout: NodeJS.Timeout) => NodeJS.Timeout,
  setCurrentActionIndex: (index: number) => void,
  setStatus: (status: ReplayStatus) => void,
  timeoutRef: TimeoutRef
): Promise<void> {
  if (statusRef.current !== ReplayStatus.PLAYING) return;

  if (index >= actionsRef.current.length) {
    setCurrentActionIndex(actionsRef.current.length - 1);
    setStatus(ReplayStatus.COMPLETED);

    window.parent.postMessage(
      {
        type: "staktrak-replay-completed",
        totalActions: actionsRef.current.length,
      },
      "*"
    );

    setTimeout(() => {
      window.parent.postMessage({ type: "staktrak-replay-fadeout" }, "*");
    }, 1000);

    if (cursorRef.current) {
      cursorRef.current.style.display = "none";
    }
    return;
  }

  const action = actionsRef.current[index];

  try {
    let element = findElement(action.selector);
    let attempts = 0;
    const maxAttempts = 5;

    while (!element && attempts < maxAttempts) {
      attempts++;
      const delay = Math.min(500 * attempts, 2000);
      await new Promise((resolve) => setTimeout(resolve, delay));
      element = findElement(action.selector);

      if (statusRef.current !== ReplayStatus.PLAYING) return;
    }

    if (!element) {
      console.warn(
        `Could not find element for action ${index}: ${action.type} on ${action.selector} after ${maxAttempts} attempts`
      );

      setCurrentActionIndex(index + 1);
      executeAction(
        index + 1,
        actionsRef,
        statusRef,
        cursorRef,
        speedRef,
        isTypingRef,
        registerTimeout,
        setCurrentActionIndex,
        setStatus,
        timeoutRef
      );
      return;
    }

    setCurrentActionIndex(index);

    window.parent.postMessage(
      {
        type: "staktrak-replay-progress",
        currentAction: index,
        totalActions: actionsRef.current.length,
        action: action,
      },
      "*"
    );

    await moveCursorToElement(element, cursorRef, statusRef);

    if (statusRef.current !== ReplayStatus.PLAYING) return;

    highlightElement(element, speedRef);

    switch (action.type) {
      case ActionType.CLICK:
        showClickEffect(cursorRef);

        element.scrollIntoView({ behavior: "smooth", block: "center" });
        await new Promise((resolve) => setTimeout(resolve, 300));

        try {
          (element as HTMLElement).focus();
        } catch (e) {
          console.warn("Could not focus element:", e);
        }

        try {
          element.dispatchEvent(
            new MouseEvent("mousedown", {
              bubbles: true,
              cancelable: true,
              view: window,
            })
          );

          await new Promise((resolve) => setTimeout(resolve, 50));

          element.dispatchEvent(
            new MouseEvent("mouseup", {
              bubbles: true,
              cancelable: true,
              view: window,
            })
          );

          await new Promise((resolve) => setTimeout(resolve, 50));

          (element as HTMLElement).click();
          element.dispatchEvent(
            new MouseEvent("click", {
              bubbles: true,
              cancelable: true,
              view: window,
            })
          );
        } catch (clickError) {
          console.error("Error during click operation:", clickError);
        }

        await new Promise((resolve) => setTimeout(resolve, 300));
        break;

      case ActionType.INPUT:
        await typeText(
          element as HTMLInputElement,
          action.value || "",
          speedRef,
          statusRef,
          isTypingRef,
          registerTimeout
        );
        break;

      case ActionType.SELECT:
        await selectOption(
          element as HTMLSelectElement,
          action.value || "",
          speedRef,
          statusRef,
          registerTimeout
        );
        break;

      case ActionType.CHECK:
        (element as HTMLInputElement).checked = true;
        element.dispatchEvent(new Event("change", { bubbles: true }));
        break;

      case ActionType.UNCHECK:
        (element as HTMLInputElement).checked = false;
        element.dispatchEvent(new Event("change", { bubbles: true }));
        break;
    }

    if (statusRef.current !== ReplayStatus.PLAYING) return;

    const nextAction = actionsRef.current[index + 1];
    let delay = MIN_DELAY / speedRef.current;

    if (nextAction && action.timestamp && nextAction.timestamp) {
      const timeDiff = nextAction.timestamp - action.timestamp;
      delay =
        Math.max(MIN_DELAY, Math.min(MAX_DELAY, timeDiff)) / speedRef.current;
    }

    timeoutRef.current = registerTimeout(
      setTimeout(() => {
        if (statusRef.current === ReplayStatus.PLAYING) {
          setCurrentActionIndex(index + 1);
          executeAction(
            index + 1,
            actionsRef,
            statusRef,
            cursorRef,
            speedRef,
            isTypingRef,
            registerTimeout,
            setCurrentActionIndex,
            setStatus,
            timeoutRef
          );
        } else {
          console.warn(
            `Not moving to next action because status is ${statusRef.current}`
          );
        }
      }, delay)
    );
  } catch (error) {
    console.error("Error executing action:", error);
    window.parent.postMessage(
      {
        type: "staktrak-replay-error",
        error: (error as Error).message,
        action: action,
      },
      "*"
    );

    if (statusRef.current === ReplayStatus.PLAYING) {
      timeoutRef.current = registerTimeout(
        setTimeout(() => {
          setCurrentActionIndex(index + 1);
          executeAction(
            index + 1,
            actionsRef,
            statusRef,
            cursorRef,
            speedRef,
            isTypingRef,
            registerTimeout,
            setCurrentActionIndex,
            setStatus,
            timeoutRef
          );
        }, 2000)
      );
    }
  }
}

export function pauseReplay(): void {
  if (statusRef.current === ReplayStatus.PLAYING) {
    statusRef.current = ReplayStatus.PAUSED;
    window.parent.postMessage({ type: "staktrak-replay-paused" }, "*");
  }
}

export function resumeReplay(): void {
  if (statusRef.current === ReplayStatus.PAUSED) {
    statusRef.current = ReplayStatus.PLAYING;

    executeAction(
      currentActionIndexRef.current,
      actionsRef,
      statusRef,
      cursorRef,
      speedRef,
      isTypingRef,
      registerTimeout,
      (index) => {
        currentActionIndexRef.current = index;
      },
      (status) => {
        statusRef.current = status;
      },
      timeoutRef
    );

    window.parent.postMessage({ type: "staktrak-replay-resumed" }, "*");
  }
}

export function stopReplay(): void {
  clearAllTimeouts();
  statusRef.current = ReplayStatus.IDLE;

  if (cursorRef.current) {
    cursorRef.current.style.display = "none";
  }

  const ripples = document.querySelectorAll(".click-ripple");
  ripples.forEach((ripple) => {
    if (ripple.parentNode) {
      ripple.parentNode.removeChild(ripple);
    }
  });

  const highlightedElements = document.querySelectorAll(".replay-pulse");
  highlightedElements.forEach((element) => {
    element.classList.remove("replay-pulse");
    (element as HTMLElement).style.outline = "";
    (element as HTMLElement).style.boxShadow = "";
    (element as HTMLElement).style.zIndex = "";
    (element as HTMLElement).style.transition = "";
  });

  window.parent.postMessage({ type: "staktrak-replay-stopped" }, "*");
}

export function initReplay(options?: ReplayOptions): void {
  const replayStyles = createReplayStyles();
  const cursor = createCursor();

  cursorRef.current = cursor;
  statusRef.current = ReplayStatus.IDLE;
  speedRef.current = DEFAULT_SPEED;
  actionsRef.current = [];
  currentActionIndexRef.current = 0;
  isTypingRef.current = false;
  timeoutIdsRef.current = [];
  timeoutRef.current = null;

  window.addEventListener("message", (event) => {
    const { data } = event;

    if (!data || !data.type) return;

    switch (data.type) {
      case "staktrak-replay-actions":
        actionsRef.current = data.actions || [];
        break;

      case "staktrak-replay-start":
        clearAllTimeouts();
        statusRef.current = ReplayStatus.PLAYING;
        currentActionIndexRef.current = 0;
        speedRef.current = data.speed || DEFAULT_SPEED;

        if (cursorRef.current) {
          cursorRef.current.style.display = "block";
        }

        setTimeout(() => {
          executeAction(
            0,
            actionsRef,
            statusRef,
            cursorRef,
            speedRef,
            isTypingRef,
            registerTimeout,
            (index) => {
              currentActionIndexRef.current = index;
            },
            (status) => {
              statusRef.current = status;
            },
            timeoutRef
          );
        }, INITIAL_DELAY / speedRef.current);
        break;

      case "staktrak-replay-pause":
        pauseReplay();
        break;

      case "staktrak-replay-resume":
        resumeReplay();
        break;

      case "staktrak-replay-stop":
        stopReplay();
        break;

      case "staktrak-replay-speed":
        speedRef.current = data.speed || DEFAULT_SPEED;
        break;

      case "staktrak-replay-ping":
        window.parent.postMessage({ type: "staktrak-replay-ready" }, "*");
        break;
    }
  });

  window.parent.postMessage({ type: "staktrak-replay-ready" }, "*");
}

export { DEFAULT_SPEED, MIN_DELAY, MAX_DELAY, INITIAL_DELAY };

document.addEventListener("DOMContentLoaded", () => {
  (window as any).convertToReplayActions = convertToReplayActions;
  (window as any).findElement = findElement;
  (window as any).pauseReplay = pauseReplay;
  (window as any).resumeReplay = resumeReplay;
  (window as any).stopReplay = stopReplay;

  initReplay();
});
