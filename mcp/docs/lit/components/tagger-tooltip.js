import { LitElement, html, css } from 'https://cdn.jsdelivr.net/gh/lit/dist@2/core/lit-core.min.js';
import { emit, fileUtils } from './utils.js';

class TaggerTooltip extends LitElement {
  static properties = {
    body: { type: String },
    file: { type: String },
    visible: { type: Boolean, reflect: true }
  };

  static styles = css`
    :host {
      position: absolute;
      display: none;
      background: #282c34;
      border-radius: 6px;
      padding: 12px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.2);
      z-index: 100;
      width: 100%;
      height: 300px;
      overflow-y: auto;
      top: -305px;
    }
    
    :host([visible]) {
      display: block;
    }
    
    pre {
      margin: 0;
      white-space: pre-wrap;
      font-family: 'Fira Code', 'Consolas', 'Monaco', monospace;
      font-size: 14px;
      line-height: 1.5;
    }
    
    code {
      display: block;
      overflow-x: auto;
      padding: 0;
    }
    
    .file-info {
      background-color: var(--tooltip-header-bg, #1e2329);
      color: var(--tooltip-header-text, #b9c0c8);
      font-size: 12px;
      padding: 4px 8px;
      margin: -12px -12px 8px -12px;
      border-bottom: 1px solid var(--tooltip-header-border, #3e4451);
      border-radius: 6px 6px 0 0;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    
    .file-path {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    
    .file-type {
      background-color: var(--tooltip-action-bg, #3e4451);
      border-radius: 4px;
      padding: 0 6px;
      margin-left: 8px;
      font-size: 11px;
      line-height: 18px;
      height: 18px;
    }
    
    .actions {
      display: flex;
      align-items: center;
    }
    
    .close-btn, .untag-btn {
      cursor: pointer;
      color: var(--close-btn-color, #b9c0c8);
      background: var(--tooltip-action-bg, #3e4451);
      border-radius: 4px;
      transition: background-color 0.2s;
    }
    
    .close-btn {
      font-size: 13px;
      font-weight: bold;
      width: 20px;
      height: 20px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .untag-btn {
      margin-right: 8px;
      padding: 0 6px;
      font-size: 11px;
      line-height: 18px;
      height: 18px;
    }
    
    .close-btn:hover, .untag-btn:hover {
      background-color: var(--tooltip-action-hover, #4e5563);
    }
  `;

  constructor() {
    super();
    this.body = '';
    this.file = '';
    this.visible = false;
  }

  render() {
    const lang = fileUtils.getHighlightJsClass(this.file);

    return html`
      <div class="file-info">
        <span class="file-path">
          ${this.file || 'Unknown file'}
          ${this.file ? html`<span class="file-type">${fileUtils.getLanguageFromFilePath(this.file)}</span>` : ''}
        </span>
        <div class="actions">
          <div class="untag-btn" @click="${this._untag}">Untag</div>
          <div class="close-btn" @click="${this._close}">×</div>
        </div>
      </div>
      <pre><code class="${lang}">${this.body || 'No content available'}</code></pre>
    `;
  }

  updated(changedProps) {
    if ((changedProps.has('body') || changedProps.has('visible')) && this.visible) {
      // Apply syntax highlighting
      setTimeout(() => {
        if (!this.visible) return; // Skip if tooltip became hidden
        
        const code = this.shadowRoot.querySelector('code');
        if (!code) return; // Skip if no code element
        
        // Only highlight if we have a valid language class
        const langClass = code.className.trim();
        
        try {
          if (typeof hljs !== 'undefined' && hljs.highlightElement) {
            hljs.highlightElement(code);
          }
        } catch (err) {
          console.error('Error highlighting code:', err);
          // Continue gracefully despite highlighting errors
        }
      }, 0);
    }
  }

  _close(e) {
    e.stopPropagation();
    this.visible = false;
  }

  _untag(e) {
    e.stopPropagation();
    emit(this, 'untag');
    this.visible = false;
  }

  show(body, file) {
    this.body = body || '';
    this.file = file || '';
    this.visible = true;
  }

  hide() {
    this.visible = false;
  }
}

customElements.define('tagger-tooltip', TaggerTooltip);