# Test Plan for PR #742: Screenshot Capture & Type Normalization

**Note:** This document covers tests specifically created for PR #742. The `src/__tests__/` directory may contain other test files.

This test plan provides comprehensive coverage for PR #742 changes:
- Type field normalization (`kind` → `type`)
- Screenshot capture with `modern-screenshot`
- Parent origin security for cross-origin iframes
- Action type renaming (`nav` → `goto`, `waitForUrl` → `waitForURL`)

## Test Files (92 test cases)

```
src/__tests__/staktrak/
├── test-helpers.ts                    # Shared utilities
├── action-type-normalization.test.ts  # ~20 tests
├── screenshot-capture.test.ts         # ~15 tests
├── parent-origin-security.test.ts     # ~12 tests
├── playwright-replay.test.ts          # ~18 tests
├── backward-compatibility.test.ts     # ~15 tests
└── e2e-screenshot-flow.test.ts        # ~12 tests
```

**Staktrak source code:** `/tests/staktrak/src/`

## Running Tests

```bash
# All staktrak tests
npx playwright test src/__tests__/staktrak/

# Specific file
npx playwright test src/__tests__/staktrak/action-type-normalization.test.ts

# UI mode
npx playwright test --ui src/__tests__/staktrak/

# Debug mode
npx playwright test --debug src/__tests__/staktrak/
```

## Import Paths

Tests import from staktrak source using relative paths:
```typescript
import { RecordingManager } from '../../../tests/staktrak/src/playwright-generator';
```

## Coverage Summary

1. **Type Normalization** - `kind` → `type`, `nav` → `goto`, `waitForURL`
2. **Screenshots** - Capture after `waitForURL`, data URL validation, error handling
3. **Security** - Parent origin capture, fallback logic, cross-origin handling
4. **Integration** - Full replay flow with screenshots
5. **Compatibility** - API unchanged, old format support
6. **E2E** - Real browser, SPA navigation, performance

## Links

- **PR:** https://github.com/stakwork/stakgraph/pull/742
- **Playwright:** https://playwright.dev/
- **modern-screenshot:** https://github.com/qq15725/modern-screenshot
