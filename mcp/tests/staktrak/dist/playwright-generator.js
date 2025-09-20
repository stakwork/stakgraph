// src/actionModel.ts
function resultsToActions(results) {
  var _a;
  const actions = [];
  const navigations = (results.pageNavigation || []).slice().sort((a, b) => a.timestamp - b.timestamp);
  const normalize = (u) => {
    var _a2;
    try {
      const url = new URL(u, ((_a2 = results.userInfo) == null ? void 0 : _a2.url) || "http://localhost");
      return url.origin + url.pathname.replace(/\/$/, "");
    } catch (e) {
      return u.replace(/[?#].*$/, "").replace(/\/$/, "");
    }
  };
  for (const nav of navigations) {
    actions.push({ kind: "nav", timestamp: nav.timestamp, url: nav.url, normalizedUrl: normalize(nav.url) });
  }
  const clicks = ((_a = results.clicks) == null ? void 0 : _a.clickDetails) || [];
  for (let i = 0; i < clicks.length; i++) {
    const cd = clicks[i];
    actions.push({
      kind: "click",
      timestamp: cd.timestamp,
      locator: {
        primary: cd.selectors.stabilizedPrimary || cd.selectors.primary,
        fallbacks: cd.selectors.fallbacks || [],
        role: cd.selectors.role,
        text: cd.selectors.text,
        tagName: cd.selectors.tagName,
        stableSelector: cd.selectors.stabilizedPrimary || cd.selectors.primary,
        candidates: cd.selectors.scores || void 0
      }
    });
    const nav = navigations.find((n) => n.timestamp > cd.timestamp && n.timestamp - cd.timestamp < 1800);
    if (nav) {
      actions.push({
        kind: "waitForUrl",
        timestamp: nav.timestamp - 1,
        // ensure ordering between click and nav
        expectedUrl: nav.url,
        normalizedUrl: normalize(nav.url),
        navRefTimestamp: nav.timestamp
      });
    }
  }
  if (results.inputChanges) {
    for (const input of results.inputChanges) {
      if (input.action === "complete" || !input.action) {
        actions.push({
          kind: "input",
          timestamp: input.timestamp,
          locator: { primary: input.elementSelector, fallbacks: [] },
          value: input.value
        });
      }
    }
  }
  if (results.formElementChanges) {
    for (const fe of results.formElementChanges) {
      actions.push({
        kind: "form",
        timestamp: fe.timestamp,
        locator: { primary: fe.elementSelector, fallbacks: [] },
        formType: fe.type,
        value: fe.value,
        checked: fe.checked
      });
    }
  }
  if (results.assertions) {
    for (const asrt of results.assertions) {
      actions.push({
        kind: "assertion",
        timestamp: asrt.timestamp,
        locator: { primary: asrt.selector, fallbacks: [] },
        value: asrt.value
      });
    }
  }
  actions.sort((a, b) => a.timestamp - b.timestamp || weightOrder(a.kind) - weightOrder(b.kind));
  refineLocators(actions);
  return actions;
}
function weightOrder(kind) {
  switch (kind) {
    case "click":
      return 1;
    case "waitForUrl":
      return 2;
    case "nav":
      return 3;
    default:
      return 4;
  }
}
function refineLocators(actions) {
  if (typeof document === "undefined")
    return;
  const seen = /* @__PURE__ */ new Set();
  for (const a of actions) {
    if (!a.locator)
      continue;
    const { primary, fallbacks } = a.locator;
    const validated = [];
    if (isUnique(primary))
      validated.push(primary);
    for (const fb of fallbacks) {
      if (validated.length >= 3)
        break;
      if (isUnique(fb))
        validated.push(fb);
    }
    if (validated.length === 0)
      continue;
    a.locator.primary = validated[0];
    a.locator.fallbacks = validated.slice(1);
    const key = a.locator.primary + "::" + a.kind;
    if (seen.has(key) && a.locator.fallbacks.length > 0) {
      a.locator.primary = a.locator.fallbacks[0];
      a.locator.fallbacks = a.locator.fallbacks.slice(1);
    }
    seen.add(a.locator.primary + "::" + a.kind);
  }
}
function isUnique(sel) {
  if (!sel || /^(html|body|div|span|p|button|input)$/i.test(sel))
    return false;
  try {
    const nodes = document.querySelectorAll(sel);
    return nodes.length === 1;
  } catch (e) {
    return false;
  }
}

// src/playwright-generator.ts
function escapeTextForAssertion(text) {
  if (!text)
    return "";
  return text.replace(/\\/g, "\\\\").replace(/'/g, "\\'").replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t").trim();
}
function normalizeText(t) {
  return (t || "").trim();
}
function locatorToSelector(l) {
  if (!l)
    return 'page.locator("body")';
  const primary = l.stableSelector || l.primary;
  if (/\[data-testid=/.test(primary)) {
    const m = primary.match(/\[data-testid=["']([^"']+)["']\]/);
    if (m)
      return `page.getByTestId('${escapeTextForAssertion(m[1])}')`;
  }
  if (primary.startsWith("#") && /^[a-zA-Z][\w-]*$/.test(primary.slice(1)))
    return `page.locator('${primary}')`;
  if (/^[a-zA-Z]+\.[a-zA-Z0-9_-]+/.test(primary)) {
    return `page.locator('${primary}')`;
  }
  if (l.role && l.text) {
    const txt = normalizeText(l.text);
    if (txt && txt.length <= 50)
      return `page.getByRole('${l.role}', { name: '${escapeTextForAssertion(txt)}' })`;
  }
  if (l.text && l.text.length <= 30 && l.text.length > 1)
    return `page.getByText('${escapeTextForAssertion(normalizeText(l.text))}')`;
  if (primary && !primary.startsWith("page."))
    return `page.locator('${primary}')`;
  for (const fb of l.fallbacks) {
    if (fb && !/^[a-zA-Z]+$/.test(fb))
      return `page.locator('${fb}')`;
  }
  return 'page.locator("body")';
}
function generatePlaywrightTestFromActions(actions, options) {
  const name = options.testName || "Recorded flow";
  const viewport = options.viewport || { width: 1280, height: 720 };
  let body = "";
  let lastTs = null;
  const base = options.baseUrl ? options.baseUrl.replace(/\/$/, "") : "";
  function fullUrl(u) {
    if (!u)
      return "";
    if (/^https?:/i.test(u))
      return u;
    if (u.startsWith("/"))
      return base + u;
    return base + "/" + u;
  }
  let i = 0;
  const collapsed = [];
  for (let k = 0; k < actions.length; k++) {
    const curr = actions[k];
    const prev = collapsed[collapsed.length - 1];
    if (curr.kind === "nav" && prev && prev.kind === "nav" && prev.url === curr.url)
      continue;
    collapsed.push(curr);
  }
  actions = collapsed;
  while (i < actions.length) {
    const a = actions[i];
    if (a.kind === "click" && i + 1 < actions.length) {
      const nxt = actions[i + 1];
      if (nxt.kind === "nav" && nxt.timestamp - a.timestamp < 1500) {
        if (lastTs != null) {
          const delta = Math.max(0, a.timestamp - lastTs);
          const wait = Math.min(3e3, Math.max(100, delta));
          if (wait > 400)
            body += `  await page.waitForTimeout(${wait});
`;
        }
        body += `  await Promise.all([
`;
        body += `    page.waitForURL('${fullUrl(nxt.url)}'),
`;
        body += `    ${locatorToSelector(a.locator)}.click()
`;
        body += `  ]);
`;
        lastTs = nxt.timestamp;
        i += 2;
        continue;
      }
    }
    if (lastTs != null) {
      const delta = Math.max(0, a.timestamp - lastTs);
      const wait = Math.min(3e3, Math.max(100, delta));
      if (wait > 500)
        body += `  await page.waitForTimeout(${wait});
`;
    }
    switch (a.kind) {
      case "nav": {
        const target = fullUrl(a.url);
        if (i === 0) {
          body += `  await page.goto('${target}');
`;
        } else {
          body += `  await page.waitForURL('${target}');
`;
        }
        break;
      }
      case "click":
        body += `  await ${locatorToSelector(a.locator)}.click();
`;
        break;
      case "input":
        body += `  await ${locatorToSelector(a.locator)}.fill('${escapeTextForAssertion(a.value || "")}');
`;
        break;
      case "form":
        if (a.formType === "checkbox" || a.formType === "radio") {
          body += a.checked ? `  await ${locatorToSelector(a.locator)}.check();
` : `  await ${locatorToSelector(a.locator)}.uncheck();
`;
        } else if (a.formType === "select") {
          body += `  await ${locatorToSelector(a.locator)}.selectOption('${escapeTextForAssertion(a.value || "")}');
`;
        }
        break;
      case "assertion":
        if (a.value && a.value.length > 0) {
          body += `  await expect(${locatorToSelector(a.locator)}).toContainText('${escapeTextForAssertion(a.value)}');
`;
        } else {
          body += `  await expect(${locatorToSelector(a.locator)}).toBeVisible();
`;
        }
        break;
    }
    lastTs = a.timestamp;
    i++;
  }
  return `import { test, expect } from '@playwright/test'

test('${name}', async ({ page }) => {
  await page.setViewportSize({ width: ${viewport.width}, height: ${viewport.height} })
${body.split("\n").filter((l) => l.trim()).map((l) => l).join("\n")}
})
`;
}
if (typeof window !== "undefined") {
  const existing = window.PlaywrightGenerator || {};
  existing.generatePlaywrightTestFromActions = generatePlaywrightTestFromActions;
  existing.generatePlaywrightTest = (baseUrl, results) => {
    try {
      const actions = resultsToActions(results);
      return generatePlaywrightTestFromActions(actions, { baseUrl });
    } catch (e) {
      console.warn("PlaywrightGenerator.generatePlaywrightTest failed", e);
      return "";
    }
  };
  window.PlaywrightGenerator = existing;
}
export {
  generatePlaywrightTestFromActions
};
