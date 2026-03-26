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

export function useSSE(url: string | null, onMessage: (update: StatusUpdate) => void) {
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  useEffect(() => {
    if (!url) return;

    let es: EventSource;
    let retryTimeout: ReturnType<typeof setTimeout>;

    function connect() {
      es = new EventSource(url!);

      es.onmessage = (e) => {
        try {
          const data: StatusUpdate = JSON.parse(e.data);
          onMessageRef.current(data);
        } catch {
          // ignore malformed events
        }
      };

      es.onerror = () => {
        es.close();
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
