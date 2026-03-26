import { useEffect, useRef } from "react";

export interface StatusUpdate {
  status: string;
  message: string;
  step: number;
  total_steps: number;
  progress: number;
  stats?: Record<string, number>;
  step_description?: string;
}

const MAX_CONSECUTIVE_ERRORS = 5;

export function useSSE(
  url: string | null,
  onMessage: (update: StatusUpdate) => void,
  onError?: (msg: string) => void,
) {
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;

  useEffect(() => {
    if (!url) return;

    let es: EventSource;
    let retryTimeout: ReturnType<typeof setTimeout>;
    let consecutiveErrors = 0;

    function connect() {
      es = new EventSource(url!);

      es.onmessage = (e) => {
        consecutiveErrors = 0;
        try {
          const data: StatusUpdate = JSON.parse(e.data);
          onMessageRef.current(data);
        } catch {
          // ignore malformed events
        }
      };

      es.onerror = () => {
        es.close();
        consecutiveErrors++;
        if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
          onErrorRef.current?.("Cannot connect to standalone server");
          return;
        }
        retryTimeout = setTimeout(connect, 3000);
      };
    }

    connect();

    return () => {
      clearTimeout(retryTimeout);
      es?.close();
    };
  }, [url]);
}
