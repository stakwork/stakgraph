import { Action, ActionLocator, resultsToActions } from "./actionModel";

function escapeTextForAssertion(text: string): string {
  if (!text) return "";
  return text
    .replace(/\\/g, "\\\\")
    .replace(/'/g, "\\'")
    .replace(/\n/g, "\\n")
    .replace(/\r/g, "\\r")
    .replace(/\t/g, "\\t")
    .trim();
}

function normalizeText(t?: string) {
  return (t || "").trim();
}

function locatorToSelector(l: ActionLocator): string {
  if (!l) return 'page.locator("body")';
  const primary = l.stableSelector || l.primary;
  
  if (/\[data-staktrak-id=/.test(primary)) {
    const m = primary.match(/\[data-staktrak-id=["']([^"']+)["']\]/);
    if (m) return `page.locator('[data-staktrak-id="${escapeTextForAssertion(m[1])}"]')`;
  }
  
  if (/\[data-testid=/.test(primary)) {
    const m = primary.match(/\[data-testid=["']([^"']+)["']\]/);
    if (m) return `page.getByTestId('${escapeTextForAssertion(m[1])}')`;
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
    if (fb && /\[data-staktrak-id=/.test(fb)) {
      const m = fb.match(/\[data-staktrak-id=["']([^"']+)["']\]/);
      if (m) return `page.locator('[data-staktrak-id="${escapeTextForAssertion(m[1])}"]')`;
    }
    if (fb && /\[data-testid=/.test(fb)) {
      const m = fb.match(/\[data-testid=["']([^"']+)["']\]/);
      if (m) return `page.getByTestId('${escapeTextForAssertion(m[1])}')`;
    }
    if (fb && !/^[a-zA-Z]+$/.test(fb)) return `page.locator('${fb}')`;
  }
  
  return 'page.locator("body")';
}

export interface GenerateOptions {
  baseUrl: string;
  viewport?: { width: number; height: number };
  testName?: string;
}

export function generatePlaywrightTestFromActions(
  actions: Action[],
  options: GenerateOptions
): string {
  const name = options.testName || "Recorded flow";
  const viewport = options.viewport || { width: 1280, height: 720 };
  let body = "";
  let lastTs: number | null = null;
  const base = options.baseUrl ? options.baseUrl.replace(/\/$/, "") : "";

  function fullUrl(u?: string) {
    if (!u) return "";
    if (/^https?:/i.test(u)) return u;
    if (u.startsWith('/')) return base + u;
    return base + '/' + u;
  }

  let i = 0;
  // collapse consecutive nav actions to the same URL
  const collapsed: Action[] = [];
  for (let k = 0; k < actions.length; k++) {
    const curr = actions[k];
    const prev = collapsed[collapsed.length - 1];
    if (curr.kind === 'nav' && prev && prev.kind === 'nav' && prev.url === curr.url) continue;
    collapsed.push(curr);
  }
  actions = collapsed;
  while (i < actions.length) {
    const a = actions[i];
    // Correlate click->nav within 1500ms
  if (a.kind === 'click' && i + 1 < actions.length) {
      const nxt = actions[i + 1];
      if (nxt.kind === 'nav' && (nxt.timestamp - a.timestamp) < 1500) {
        // optional pre-wait before click if gap from previous
        if (lastTs != null) {
          const delta = Math.max(0, a.timestamp - lastTs);
            const wait = Math.min(3000, Math.max(100, delta));
            if (wait > 400) body += `  await page.waitForTimeout(${wait});\n`;
        }
        body += `  await Promise.all([\n`;
        body += `    page.waitForURL('${fullUrl(nxt.url)}'),\n`;
        body += `    ${locatorToSelector(a.locator!)}.click()\n`;
        body += `  ]);\n`;
        lastTs = nxt.timestamp;
        i += 2;
        continue;
      }
    }

    if (lastTs != null) {
      const delta = Math.max(0, a.timestamp - lastTs);
      const wait = Math.min(3000, Math.max(100, delta));
      if (wait > 500) body += `  await page.waitForTimeout(${wait});\n`;
    }

    switch (a.kind) {
      case 'nav': {
        const target = fullUrl(a.url);
        if (i === 0) {
          body += `  await page.goto('${target}');\n`;
        } else {
          body += `  await page.waitForURL('${target}');\n`;
        }
        break;
      }
      case 'click':
        body += `  await ${locatorToSelector(a.locator!)}.click();\n`;
        break;
      case 'input':
        body += `  await ${locatorToSelector(a.locator!)}.fill('${escapeTextForAssertion(a.value || '')}');\n`;
        break;
      case 'form':
        if (a.formType === 'checkbox' || a.formType === 'radio') {
          body += a.checked
            ? `  await ${locatorToSelector(a.locator!)}.check();\n`
            : `  await ${locatorToSelector(a.locator!)}.uncheck();\n`;
        } else if (a.formType === 'select') {
          body += `  await ${locatorToSelector(a.locator!)}.selectOption('${escapeTextForAssertion(a.value || '')}');\n`;
        }
        break;
      case 'assertion':
        if (a.value && a.value.length > 0) {
          body += `  await expect(${locatorToSelector(a.locator!)}).toContainText('${escapeTextForAssertion(a.value)}');\n`;
        } else {
          body += `  await expect(${locatorToSelector(a.locator!)}).toBeVisible();\n`;
        }
        break;
    }
    lastTs = a.timestamp;
    i++;
  }
  return `import { test, expect } from '@playwright/test'

test('${name}', async ({ page }) => {
  await page.setViewportSize({ width: ${viewport.width}, height: ${viewport.height} })
${body
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => l)
  .join("\n")}
})
`;
}

if (typeof window !== "undefined") {
  const existing = (window as any).PlaywrightGenerator || {};
  existing.generatePlaywrightTestFromActions = generatePlaywrightTestFromActions;
  existing.locatorToSelector = locatorToSelector;
  existing.generatePlaywrightTest = (baseUrl: string, results: any) => {
    try {
      const actions = resultsToActions(results);
      return generatePlaywrightTestFromActions(actions, { baseUrl });
    } catch (e) {
      console.warn('PlaywrightGenerator.generatePlaywrightTest failed', e);
      return '';
    }
  };
  (window as any).PlaywrightGenerator = existing;
}
