/**
 * Action management utilities for StakTrak
 * Provides centralized action types and helper functions
 */

export interface Action {
  id: string;
  type: 'goto' | 'click' | 'input' | 'form' | 'assertion' | 'waitForURL';
  timestamp: number;
  [key: string]: any;
}

export interface NavigationAction extends Action {
  type: 'goto';
  url: string;
}

export interface ClickAction extends Action {
  type: 'click';
  locator?: {
    text?: string;
    primary?: string;
  };
}

export interface InputAction extends Action {
  type: 'input';
  value: string;
}

export interface FormAction extends Action {
  type: 'form';
  formType?: 'checkbox' | 'radio' | 'select';
  checked?: boolean;
  value?: string;
}

export interface AssertionAction extends Action {
  type: 'assertion';
  value: string;
  selector: string;
}

export interface WaitForUrlAction extends Action {
  type: 'waitForURL';
  expectedUrl?: string;
}

export type AnyAction = NavigationAction | ClickAction | InputAction | FormAction | AssertionAction | WaitForUrlAction;

/**
 * Maps action types to their corresponding message types
 */
export const ACTION_MESSAGE_MAP: Record<string, string> = {
  'goto': 'staktrak-remove-navigation',
  'click': 'staktrak-remove-click',
  'input': 'staktrak-remove-input',
  'form': 'staktrak-remove-form',
  'assertion': 'staktrak-remove-assertion'
};

/**
 * Get display text for an action
 */
export function getActionDisplayText(action: AnyAction): string {
  switch (action.type) {
    case 'goto':
      return `Navigate to ${(action as NavigationAction).url || '/'}`;
    case 'click':
      const click = action as ClickAction;
      const clickText = click.locator?.text ? `"${click.locator.text}"` : click.locator?.primary || 'element';
      return `Click ${clickText}`;
    case 'input':
      const inputValue = (action as InputAction).value;
      return `Type "${inputValue.length > 30 ? inputValue.substring(0, 30) + '...' : inputValue}"`;
    case 'form':
      const form = action as FormAction;
      if (form.formType === 'checkbox' || form.formType === 'radio') {
        return `${form.checked ? 'Check' : 'Uncheck'} ${form.formType}`;
      } else if (form.formType === 'select') {
        return `Select "${form.value}"`;
      }
      return `Form: ${form.value}`;
    case 'assertion':
      const assertValue = (action as AssertionAction).value;
      return `Assert "${assertValue.length > 30 ? assertValue.substring(0, 30) + '...' : assertValue}"`;
    case 'waitForURL':
      return `Wait for ${(action as WaitForUrlAction).expectedUrl || 'navigation'}`;
    default:
      return action.type;
  }
}

