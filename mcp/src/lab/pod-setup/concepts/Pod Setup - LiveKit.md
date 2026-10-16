---
description: "LiveKit in a pod: the LiveKit server in dev mode beside the app, with fixed dev keys."
parent: Pod Setup - Local Services
---
LiveKit in a pod: the LiveKit server in dev mode beside the app, with fixed dev keys.

- A `livekit` compose service (`image: livekit/livekit-server`, `command:
  --dev --bind 0.0.0.0`), `ports: ["7880:7880", "7881:7881"]`.
- The app's env: `LIVEKIT_URL` is `ws://localhost:7880`, `LIVEKIT_API_KEY`
  `devkey`, `LIVEKIT_API_SECRET`
  `devsecret-change-in-production-12345678901234`.
