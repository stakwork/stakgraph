import { LitElement, html, css } from 'https://cdn.jsdelivr.net/gh/lit/dist@2/core/lit-core.min.js';
import { emit, constants } from './utils.js';

class TaggerEditor extends LitElement {
  static properties = {
    activeTagElement: { type: Object, state: true },
    currentTag: { type: String, state: true },
    currentTrigger: { type: String, state: true }
  };

  static styles = css`
    :host {
      flex: 1;
      position: relative;
    }
    
    #editor {
      border: 1px solid var(--border-color, #cccccc);
      min-height: 180px;
      padding: 15px;
      outline: none;
      line-height: 1.6;
      font-size: 16px;
      border-radius: 5px 0 0 5px;
      box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
      background-color: var(--editor-bg, #ffffff);
      color: var(--text-color, #000000);
    }
    
    .tagged-text {
      font-weight: bold;
      cursor: pointer;
      border-radius: 3px;
      padding: 1px 2px;
    }
  `;

  constructor() {
    super();
    this.activeTagElement = null;
    this.lastRange = null;
    this.currentTag = '';
    this.currentTrigger = '';
    this.searchTimeout = null;
    this.previousContent = '';
  }

  render() {
    return html`
      <div id="editor" contenteditable="true" 
           @input="${this._handleInput}"
           @keydown="${this._handleKeydown}"
           @keyup="${this._handleKeyup}"
           @click="${this._handleClick}">
      </div>
    `;
  }

  firstUpdated() {
    // Focus editor on load
    this.editor.focus();
  }

  get editor() {
    return this.shadowRoot.querySelector('#editor');
  }

  _handleInput(e) {
    // Clean up empty spans if needed
    if (this._cleanupEmptyTaggedSpans()) return;
    
    clearTimeout(this.searchTimeout);
    this.searchTimeout = setTimeout(() => {
      // Get the current text content
      const currentContent = this.editor.textContent;
      
      // Check for trigger characters
      const triggerResult = this._checkForTrigger(currentContent);
      
      if (triggerResult) {
        this.currentTag = triggerResult.text;
        this.currentTrigger = triggerResult.trigger;
        // Save current selection
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
          this.lastRange = selection.getRangeAt(0).cloneRange();
        }
        
        console.log('Trigger found:', triggerResult);
        emit(this, 'tag-input', { tag: triggerResult.text, trigger: triggerResult.trigger });
      } else {
        emit(this, 'tag-input-end');
      }
      
      // Update previous content
      this.previousContent = currentContent;
    }, 200);
  }

  // New method to check for trigger characters in text
  _checkForTrigger(content) {
    if (!content) return null;
    
    // Get current cursor position
    const selection = window.getSelection();
    if (!selection.rangeCount) return null;
    
    const cursorPos = this._getCursorPosition();
    if (cursorPos === null) return null;
    
    // Look backward from cursor position to find a trigger character
    let i = cursorPos - 1;
    let triggerPos = -1;
    let triggerChar = '';
    
    while (i >= 0) {
      const char = content.charAt(i);
      
      if (char === '#' || char === '@' || char === '/') {
        triggerPos = i;
        triggerChar = char;
        break;
      } else if (!/[\w\.-]/.test(char)) {
        // If we hit a non-word character that's not part of the tag, stop searching
        break;
      }
      
      i--;
    }
    
    if (triggerPos === -1) return null;
    
    // Extract the tag text (without the trigger character)
    const tagText = content.substring(triggerPos + 1, cursorPos);
    
    // Validate the tag text
    return (tagText && /^[\w\.-]+$/.test(tagText)) ? 
      { text: tagText, trigger: triggerChar } : null;
  }

  // Helper method to get cursor position in text
  _getCursorPosition() {
    const selection = window.getSelection();
    if (!selection.rangeCount) return null;
    
    const range = selection.getRangeAt(0);
    
    // Get all text nodes in the editor
    const textNodes = [];
    const walker = document.createTreeWalker(
      this.editor,
      NodeFilter.SHOW_TEXT,
      null,
      false
    );
    
    let node;
    while (node = walker.nextNode()) {
      textNodes.push(node);
    }
    
    if (textNodes.length === 0) return 0;
    
    // Calculate cursor position
    let pos = 0;
    for (let i = 0; i < textNodes.length; i++) {
      const node = textNodes[i];
      
      if (node === range.startContainer) {
        return pos + range.startOffset;
      }
      
      pos += node.length;
    }
    
    return pos;
  }

  _handleKeydown(e) {
    // Navigation keys
    if (e.key === 'Escape') {
      emit(this, 'escape-key');
      return;
    }
    
    // Pass arrow keys for result navigation
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      emit(this, 'navigate-down');
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      emit(this, 'navigate-up');
    } else if (e.key === 'Enter') {
      e.preventDefault();
      emit(this, 'enter-key');
    } else if (e.key === 'Shift') {
      emit(this, 'shift-key', { pressed: true });
    }
  }

  _handleKeyup(e) {
    if (e.key === 'Backspace' || e.key === 'Delete') {
      this._cleanupEmptyTaggedSpans();
    } else if (e.key === 'Shift') {
      emit(this, 'shift-key', { pressed: false });
    }
  }

  _handleClick(e) {
    const tagElement = e.target.closest('.tagged-text');
    
    // If clicking on a tag element
    if (tagElement) {
      // Don't trigger if already active
      if (tagElement === this.activeTagElement) return;
      
      this.activeTagElement = tagElement;
      emit(this, 'tag-click', { 
        body: tagElement.dataset.body, 
        file: tagElement.dataset.file 
      });
    } else {
      // Clicked elsewhere in editor
      this.activeTagElement = null;
      emit(this, 'editor-click');
    }
  }

  // Remove empty spans that might have been created
  _cleanupEmptyTaggedSpans() {
    const editor = this.editor;
    const rawContent = editor.innerHTML;
    const textContent = editor.textContent.trim();
    
    if (textContent === '' || 
        rawContent === '' || 
        rawContent === '<br>' || 
        rawContent === '&nbsp;' || 
        /^(\s|&nbsp;|<br\s*\/?>|<div>(<br\s*\/?>)*<\/div>)*$/i.test(rawContent)) {
      
      editor.innerHTML = '';
      
      // Reset state
      this.currentTag = '';
      this.lastRange = null;
      this.activeTagElement = null;
      this.currentTrigger = '';
      
      // Reset cursor position
      const range = document.createRange();
      range.setStart(editor, 0);
      range.collapse(true);
      
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      
      return true;
    }
    return false;
  }

  // Replace the tag with a formatted span
  replaceTagWithStyledSpan(result) {
    if (!this.lastRange) {
      console.error('No range saved for tag replacement');
      return;
    }
    
    try {
      // Get current selection and restore our saved range
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(this.lastRange);
      
      // Get text content and cursor position
      const currentContent = this.editor.textContent;
      const cursorPos = this._getCursorPosition();
      
      if (cursorPos === null) {
        console.error('Could not determine cursor position');
        return;
      }
      
      // Find trigger start by searching backward from cursor
      let triggerPos = -1;
      for (let i = cursorPos - 1; i >= 0; i--) {
        const char = currentContent.charAt(i);
        if (char === '#' || char === '/' || char === '@') {
          triggerPos = i;
          break;
        } else if (!/[\w\.-]/.test(char)) {
          break;
        }
      }
      
      if (triggerPos === -1) {
        console.error('Could not find trigger position');
        return;
      }
      
      // Get range from trigger to cursor
      const range = this.lastRange.cloneRange();
      
      // Create the replacement span element
      const color = constants.nodeTypeColors[result.node_type] || '#333';
      const tagSpan = document.createElement('span');
      tagSpan.className = 'tagged-text';
      tagSpan.style.color = color;
      
      // Use filename for File node type
      if (result.node_type === 'File' && result.properties.name) {
        tagSpan.textContent = `@${result.properties.name}`;
      } else {
        tagSpan.textContent = `${this.currentTrigger}${this.currentTag}`;
      }
      
      // Store metadata
      tagSpan.dataset.body = result.properties.body || '';
      tagSpan.dataset.file = result.properties.file || '';
      tagSpan.dataset.nodeType = result.node_type;
      
      // Execute command to delete the current tag text
      document.execCommand('delete', false);
      
      // Insert the styled span
      document.execCommand('insertHTML', false, tagSpan.outerHTML + '&nbsp;');
      
      // Reset state
      this.currentTag = '';
      this.lastRange = null;
    } catch (error) {
      console.error('Error replacing tag with span:', error);
    }
  }

  // Untag the active element
  untagActiveElement() {
    if (!this.activeTagElement) return;
    
    // Create text node with tag content
    const textNode = document.createTextNode(this.activeTagElement.textContent);
    
    // Replace the tag with plain text
    this.activeTagElement.parentNode.replaceChild(textNode, this.activeTagElement);
    
    this.activeTagElement = null;
  }
}

customElements.define('tagger-editor', TaggerEditor);