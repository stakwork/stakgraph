import {
  pressBack,
  pressHome,
  swipe,
  takeScreenshot,
  tapByCoordinates,
  tapBySelector,
  typeIntoElement,
} from "./appium";
import {
  AndroidSelector,
  RecordedAction,
  ReplayEvent,
  ReplaySummary,
} from "./types";

async function runAction(action: RecordedAction): Promise<void> {
  switch (action.type) {
    case "tap":
      if (action.selector) {
        await tapBySelector(action.selector as AndroidSelector);
      } else {
        if (action.x === undefined || action.y === undefined) {
          throw new Error("Tap action requires selector or x/y coordinates.");
        }
        await tapByCoordinates(action.x, action.y);
      }
      return;
    case "type":
      await typeIntoElement(action.selector, action.text, action.replace);
      return;
    case "swipe":
      await swipe(action.startX, action.startY, action.endX, action.endY, action.durationMs);
      return;
    case "back":
      await pressBack();
      return;
    case "home":
      await pressHome();
      return;
    default:
      throw new Error(`Unsupported action type: ${(action as { type: string }).type}`);
  }
}

export async function replayActions(
  actions: RecordedAction[],
  onEvent: (event: ReplayEvent) => void
): Promise<ReplaySummary> {
  const errors: Array<{ index: number; message: string }> = [];
  const total = actions.length;
  let completed = 0;

  onEvent({ type: "started", total });

  for (let index = 0; index < actions.length; index++) {
    const action = actions[index];

    try {
      await runAction(action);
      const screenshot = await takeScreenshot();

      completed += 1;
      onEvent({
        type: "progress",
        current: index + 1,
        total,
        action,
        screenshot,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown replay error";
      errors.push({ index, message });

      let screenshot: string | undefined;
      try {
        screenshot = await takeScreenshot();
      } catch {
        screenshot = undefined;
      }

      onEvent({
        type: "error",
        current: index + 1,
        total,
        action,
        error: message,
        screenshot,
      });
    }
  }

  onEvent({ type: "completed", total, errors: errors.length });

  return {
    total,
    completed,
    errors,
  };
}