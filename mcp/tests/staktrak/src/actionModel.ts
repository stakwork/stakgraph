import { Results } from './types'

export type ActionKind = 'click' | 'input' | 'form' | 'assertion' | 'nav'

export interface ActionLocator {
  primary: string
  fallbacks: string[]
  role?: string
  text?: string
  tagName?: string
}

export interface Action {
  kind: ActionKind
  timestamp: number
  locator?: ActionLocator
  value?: string
  checked?: boolean
  formType?: string
  url?: string
}

export function resultsToActions(results: Results): Action[] {
  const actions: Action[] = []

  if (results.pageNavigation) {
    for (const nav of results.pageNavigation) {
      actions.push({ kind: 'nav', timestamp: nav.timestamp, url: nav.url })
    }
  }

  if (results.clicks?.clickDetails) {
    for (const cd of results.clicks.clickDetails) {
      actions.push({
        kind: 'click',
        timestamp: cd.timestamp,
        locator: {
          primary: cd.selectors.primary,
            fallbacks: cd.selectors.fallbacks || [],
            role: cd.selectors.role,
            text: cd.selectors.text,
            tagName: cd.selectors.tagName
        }
      })
    }
  }

  if (results.inputChanges) {
    for (const input of results.inputChanges) {
      if (input.action === 'complete' || !input.action) {
        actions.push({
          kind: 'input',
          timestamp: input.timestamp,
          locator: { primary: input.elementSelector, fallbacks: [] },
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
        locator: { primary: fe.elementSelector, fallbacks: [] },
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
        locator: { primary: asrt.selector, fallbacks: [] },
        value: asrt.value
      })
    }
  }

  actions.sort((a, b) => a.timestamp - b.timestamp)
  refineLocators(actions)
  return actions
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
