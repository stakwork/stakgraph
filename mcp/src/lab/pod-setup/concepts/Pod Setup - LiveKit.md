---
description: "LiveKit in a pod: the LiveKit server in dev mode beside the app, with fixed dev keys."
parent: Pod Setup - Local Services
---
LiveKit in a pod: the LiveKit server in dev mode beside the app, with fixed dev keys.

- A `livekit` compose service (`image: livekit/livekit-server`, `command:
  --dev --bind 0.0.0.0`), `ports: ["7880:7880", "7881:7881",
  "7882:7882/udp"]` — 7880 the API and signalling, 7881 (TCP) and 7882
  (UDP) the media, which never flows without them — and `LIVEKIT_KEYS:
  "devkey: devsecret-change-in-production-12345678901234"`. Without
  `LIVEKIT_KEYS`, dev mode's key is `devkey: secret`, and the server
  refuses every token the app signs with any other secret.
- The app's env: `LIVEKIT_URL` is `ws://127.0.0.1:7880`, `LIVEKIT_API_KEY`
  `devkey`, `LIVEKIT_API_SECRET`
  `devsecret-change-in-production-12345678901234` — the pair in
  `LIVEKIT_KEYS`.
