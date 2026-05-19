// ErrorBoundary wraps an arbitrary subtree and surfaces any thrown
// error as a banner instead of crashing the SPA. Preact's Component
// supports `componentDidCatch` exactly like React.
//
// Phase 8 mainly catches "rendered uPlot with no points" / "missing
// shape after a backend change" — Query errors flow through
// Tanstack's own error states, not through here.

import { Component } from "preact";
import type { ComponentChildren } from "preact";

interface Props {
  children: ComponentChildren;
}
interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(err: Error): State {
    return { error: err };
  }

  componentDidCatch(err: Error) {
    // eslint-disable-next-line no-console
    console.error("ErrorBoundary caught:", err);
  }

  render() {
    if (this.state.error) {
      return (
        <div class="error-banner">
          Something broke rendering this view: {this.state.error.message}
        </div>
      );
    }
    return this.props.children;
  }
}
