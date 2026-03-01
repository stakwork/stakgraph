import {
  RecordedAction,
  RecordedActionInput,
  RecordingContext,
} from "./types";

class Recorder {
  private active = false;
  private actions: RecordedAction[] = [];
  private startedAt = 0;
  private context: RecordingContext | null = null;

  start(context: RecordingContext): void {
    this.active = true;
    this.actions = [];
    this.startedAt = Date.now();
    this.context = context;
  }

  stop(): { actions: RecordedAction[]; startedAt: number; stoppedAt: number; context: RecordingContext | null } {
    const payload = {
      actions: [...this.actions],
      startedAt: this.startedAt,
      stoppedAt: Date.now(),
      context: this.context,
    };

    this.active = false;
    this.actions = [];
    this.startedAt = 0;
    this.context = null;
    return payload;
  }

  isActive(): boolean {
    return this.active;
  }

  getContext(): RecordingContext | null {
    return this.context;
  }

  getActions(): RecordedAction[] {
    return [...this.actions];
  }

  record(action: RecordedActionInput): void {
    if (!this.active) {
      return;
    }

    this.actions.push({
      ...action,
      timestamp: Date.now(),
    } as RecordedAction);
  }
}

export const recorder = new Recorder();