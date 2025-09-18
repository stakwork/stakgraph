import { Results } from './types'

export type ActionKind = 'click' | 'input' | 'form' | 'assertion' | 'nav' | 'waitForUrl'

export interface ActionLocator {
  primary: string
  fallbacks: string[]
  role?: string
  text?: string
  tagName?: string
  stableSelector?: string
  candidates?: { selector: string; score: number; reasons: string[] }[]
}

export interface Action {
  kind: ActionKind
  timestamp: number
  locator?: ActionLocator
  value?: string
  checked?: boolean
  formType?: string
  url?: string
  expectedUrl?: string
  normalizedUrl?: string
  navRefTimestamp?: number
}

export function resultsToActions(results: Results): Action[] {
  const actions: Action[] = []
  const navigations = (results.pageNavigation || []).slice().sort((a,b)=>a.timestamp-b.timestamp)

  // Helper to normalize URLs (strip query & hash, remove trailing slash)
  const normalize = (u: string): string => {
    const getBaseUrl = (): string => {
      if (results.userInfo?.url) return results.userInfo.url;
      if (typeof window !== 'undefined' && window.location) return window.location.href;
      return 'http://localhost:3000';
    };
    try { 
      const url = new URL(u, getBaseUrl()); 
      return url.origin + url.pathname.replace(/\/$/,''); 
    } catch { 
      return u.replace(/[?#].*$/,'').replace(/\/$/,'') 
    }
  }

  for (const nav of navigations) {
    actions.push({ kind: 'nav', timestamp: nav.timestamp, url: nav.url, normalizedUrl: normalize(nav.url) })
  }

  const clicks = results.clicks?.clickDetails || []
  for (let i=0;i<clicks.length;i++) {
    const cd = clicks[i]
    actions.push({
      kind: 'click',
      timestamp: cd.timestamp,
      locator: {
        primary: cd.selectors.stabilizedPrimary || cd.selectors.primary,
        fallbacks: cd.selectors.fallbacks || [],
        role: cd.selectors.role,
        text: cd.selectors.text,
        tagName: cd.selectors.tagName,
        stableSelector: cd.selectors.stabilizedPrimary || cd.selectors.primary,
        candidates: (cd.selectors as any).scores || undefined
      }
    })
    // Find the first navigation within 1800ms after this click
    const nav = navigations.find(n => n.timestamp > cd.timestamp && n.timestamp - cd.timestamp < 1800)
    if (nav) {
      actions.push({
        kind: 'waitForUrl',
        timestamp: nav.timestamp - 1, // ensure ordering between click and nav
        expectedUrl: nav.url,
        normalizedUrl: normalize(nav.url),
        navRefTimestamp: nav.timestamp
      })
    }
  }

  if (results.inputChanges) {
    for (const input of results.inputChanges) {
      if (input.action === 'complete' || !input.action) {
        actions.push({
          kind: 'input',
          timestamp: input.timestamp,
          locator: { primary: input.elementSelector, stableSelector: input.elementSelector, fallbacks: [] },
          value: input.value
        })
      }
    }
  }

  if (results.formElementChanges) {
    for (const fe of results.formElementChanges) {
      actions.push({
        kind: 'form',
        timestamp: fe.timestamp,
        locator: { primary: fe.elementSelector, stableSelector: fe.elementSelector, fallbacks: [] },
        formType: fe.type,
        value: fe.value,
        checked: fe.checked
      })
    }
  }

  if (results.assertions) {
    for (const asrt of results.assertions) {
      actions.push({
        kind: 'assertion',
        timestamp: asrt.timestamp,
        locator: { primary: asrt.selector, stableSelector: asrt.selector, fallbacks: [] },
        value: asrt.value
      })
    }
  }

  actions.sort((a, b) => a.timestamp - b.timestamp || weightOrder(a.kind)-weightOrder(b.kind))
  refineLocators(actions)
  return actions
}

function weightOrder(kind: ActionKind): number {
  switch(kind){
    case 'click': return 1
    case 'waitForUrl': return 2
    case 'nav': return 3
    default: return 4
  }
}

function refineLocators(actions: Action[]) {
  if (typeof document === 'undefined') return
  const seen = new Set<string>()
  for (const a of actions) {
    if (!a.locator) continue
    const { primary, fallbacks } = a.locator
    const validated: string[] = []
    if (isUnique(primary)) validated.push(primary)
    for (const fb of fallbacks) {
      if (validated.length >= 3) break
      if (isUnique(fb)) validated.push(fb)
    }
    if (validated.length === 0) continue
    a.locator.primary = validated[0]
    a.locator.fallbacks = validated.slice(1)
    const key = a.locator.primary + '::' + a.kind
    if (seen.has(key) && a.locator.fallbacks.length > 0) {
      a.locator.primary = a.locator.fallbacks[0]
      a.locator.fallbacks = a.locator.fallbacks.slice(1)
    }
    seen.add(a.locator.primary + '::' + a.kind)
  }
}

function isUnique(sel: string): boolean {
  if (!sel || /^(html|body|div|span|p|button|input)$/i.test(sel)) return false
  try {
    const nodes = document.querySelectorAll(sel)
    return nodes.length === 1
  } catch {
    return false
  }
}
