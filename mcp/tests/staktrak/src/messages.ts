/**
 * Message type definitions for StakTrak communication
 */

export interface BaseMessage {
  type: string;
}

export interface StartMessage extends BaseMessage {
  type: 'staktrak-start';
}

export interface StopMessage extends BaseMessage {
  type: 'staktrak-stop';
}

export interface EnableSelectionMessage extends BaseMessage {
  type: 'staktrak-enable-selection';
}

export interface DisableSelectionMessage extends BaseMessage {
  type: 'staktrak-disable-selection';
}

export interface AddAssertionMessage extends BaseMessage {
  type: 'staktrak-add-assertion';
  assertion: {
    id: string;
    type?: string;
    selector: string;
    value?: string;
    timestamp?: number;
  };
}

export interface RemoveAssertionMessage extends BaseMessage {
  type: 'staktrak-remove-assertion';
  assertionId: string;
}

export interface ClearAssertionsMessage extends BaseMessage {
  type: 'staktrak-clear-assertions';
}

export interface ClearAllActionsMessage extends BaseMessage {
  type: 'staktrak-clear-all-actions';
}

export interface RemoveActionMessage extends BaseMessage {
  type: 'staktrak-remove-navigation' | 'staktrak-remove-click' |
        'staktrak-remove-input' | 'staktrak-remove-form';
  timestamp: number;
  actionId?: string;
}

export interface DebugRequestMessage extends BaseMessage {
  type: 'staktrak-debug-request';
  messageId: string;
  coordinates?: { x: number; y: number };
}

export interface RecoverMessage extends BaseMessage {
  type: 'staktrak-recover';
}

export interface SelectionMessage extends BaseMessage {
  type: 'staktrak-selection';
  text: string;
  selector: string;
  assertionId?: string;
}

export interface ResultsMessage extends BaseMessage {
  type: 'staktrak-results';
  data: any; // TrackingResults type from types.ts
}

export interface ActionAddedMessage extends BaseMessage {
  type: 'staktrak-action-added';
  action: {
    id: string;
    kind: 'click' | 'form' | 'nav' | 'input' | 'assertion';
    timestamp: number;
    [key: string]: any;
  };
}

export interface PopupMessage extends BaseMessage {
  type: 'staktrak-popup';
  message: string;
  popupType?: 'success' | 'error' | 'info';
}

export type StakTrakMessage =
  | StartMessage
  | StopMessage
  | EnableSelectionMessage
  | DisableSelectionMessage
  | AddAssertionMessage
  | RemoveAssertionMessage
  | ClearAssertionsMessage
  | ClearAllActionsMessage
  | RemoveActionMessage
  | DebugRequestMessage
  | RecoverMessage
  | SelectionMessage
  | ResultsMessage
  | ActionAddedMessage
  | PopupMessage;

/**
 * Type guard to check if a message is a StakTrak message
 */
export function isStakTrakMessage(event: MessageEvent): event is MessageEvent & { data: StakTrakMessage } {
  return event.data &&
         typeof event.data.type === 'string' &&
         event.data.type.startsWith('staktrak-');
}

/**
 * Type guard for specific message types
 */
export function isRemoveActionMessage(message: any): message is RemoveActionMessage {
  return message.type &&
         message.type.startsWith('staktrak-remove-') &&
         typeof message.timestamp === 'number';
}

export function isAddAssertionMessage(message: any): message is AddAssertionMessage {
  return message.type === 'staktrak-add-assertion' &&
         message.assertion &&
         typeof message.assertion.id === 'string';
}