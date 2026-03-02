# staktrak-android

`staktrak-android` is a local automation bridge for Android apps.

It exposes a simple HTTP API that an AI agent can call to:

- inspect the current Android UI tree,
- perform UI actions (tap/type/swipe/back/home),
- record those actions into a structured timeline,
- generate executable Appium JavaScript from that timeline,
- replay recorded/generated flows while streaming step-by-step progress.

In short: this service turns Android UI automation into an agent-friendly API workflow instead of requiring the agent to manage raw Appium commands directly.

## Agent-facing intent

This project is designed for agent loops such as:

1. Start session for a target package/activity.
2. Read `/tree` to discover controls and selectors.
3. Execute actions via `/tap`, `/type`, `/swipe`, etc.
4. Stop session to receive recorded `actions` + generated `script`.
5. Replay via `/session/replay` and observe `/events` SSE progress.

## What this service provides

- Local HTTP API for Android UI automation actions.
- Session lifecycle management (start/stop/replay).
- Accessibility tree extraction from Appium page source XML.
- Action recording and script generation (Appium JS output).
- Replay progress streaming via Server-Sent Events.
- Request validation with Zod on key payload endpoints.

## Defaults

- Service port: `4724`
- Appium server URL: `http://127.0.0.1:4723`
- Appium base path default: `/`

## Install and run

```bash
npm install
npm run build
npm run dev
# or
npm start
```

Service starts at:

- `http://localhost:4724`

## Required runtime app target

When starting a session, the service must know which Android app to attach to.

`POST /session/start` supports runtime fields:

- `package` (required unless env fallback is set)
- `activity` (optional)
- `deviceName` (optional)

### Example

```json
{
  "package": "com.example.app",
  "activity": ".MainActivity",
  "deviceName": "Android"
}
```

### Env fallbacks

- `APPIUM_SERVER_URL`
- `APPIUM_APP_PACKAGE`
- `APPIUM_APP_ACTIVITY`
- `APPIUM_DEVICE_NAME`
- `PORT`

If `package` is not sent and `APPIUM_APP_PACKAGE` is missing, session start fails with `400`.

## Endpoints

### Basic

- `GET /health` → `{ "ok": true }`
- `GET /session` → current session metadata + recording status
- `GET /events` → SSE stream for replay events

### Device tree and actions

- `GET /tree` → raw XML + parsed clickable/text elements
- `POST /tap` → tap by selector or coordinates
- `POST /type` → type text into selector target
- `POST /swipe` → swipe gesture
- `POST /screenshot` → base64 screenshot
- `POST /back` → Android back key
- `POST /home` → Android home key

### Session flow

- `POST /session/start` → start Appium session and begin recording
- `POST /session/stop` → stop recording/session and return:
  - `actions` (recorded action JSON)
  - `script` (generated Appium JS)
- `POST /session/replay` → replay from `actions` or generated `script`

## Selector format

Selector priority is:

1. `resourceId`
2. `accessibilityId`
3. `text`
4. `xpath`

Example selector:

```json
{
  "resourceId": "com.example:id/login_button"
}
```

## Replay events (SSE)

`GET /events` emits replay lifecycle messages as `event: replay`.

Event shapes include:

- `started`
- `progress` (includes current action + screenshot)
- `error` (includes action + error + optional screenshot)
- `completed`

## Validation behavior

Zod validation is applied to major request payloads (`/tap`, `/type`, `/swipe`, `/session/start`, `/session/replay`).

On validation errors:

- HTTP `400`
- Body includes:
  - `error`
  - `details` (path/message list)

## Quick curl examples

### Health

```bash
curl http://localhost:4724/health
```

### Start session

```bash
curl -X POST http://localhost:4724/session/start \
  -H "content-type: application/json" \
  -d '{"package":"com.example.app"}'
```

### Tap by coordinates

```bash
curl -X POST http://localhost:4724/tap \
  -H "content-type: application/json" \
  -d '{"x":540,"y":1200}'
```

### Stop session and generate script

```bash
curl -X POST http://localhost:4724/session/stop
```
