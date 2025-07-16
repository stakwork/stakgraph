import { useState, useEffect } from "https://esm.sh/preact/hooks";

export function useMessageListener() {
  const [showControls, setShowControls] = useState(false);
  useEffect(() => {
    const handleMessage = (event) => {
      if (event.data && event.data.type) {
        switch (event.data.type) {
          case "staktrak-setup":
            setShowControls(true);
            break;
          // Add more cases as needed
          default:
            break;
        }
      }
    };

    window.addEventListener("message", handleMessage);

    // Cleanup function to remove event listener
    return () => {
      window.removeEventListener("message", handleMessage);
    };
  }, []);

  return { showControls };
}
