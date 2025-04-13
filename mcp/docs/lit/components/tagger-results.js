import { LitElement, html, css } from 'https://cdn.jsdelivr.net/gh/lit/dist@2/core/lit-core.min.js';
import { emit, constants, fileUtils } from './utils.js';

class TaggerResults extends LitElement {
  static properties = {
    results: { type: Array },
    visible: { type: Boolean, reflect: true },
    selectedIndex: { type: Number }
  };

  static styles = css`
    :host {
      width: 150px;
      border: 1px solid var(--border-color, #cccccc);
      border-left: none;
      border-radius: 0 5px 5px 0;
      padding: 10px;
      display: none;
      overflow-y: auto;
      background-color: var(--results-bg, #f9f9f9);
      height: 180px;
      color: var(--text-color, #000000);
    }
    
    :host([visible]) {
      display: block;
    }
    
    h3 {
      margin-top: 0;
      font-size: 14px;
      margin-bottom: 10px;
      text-align: center;
    }
    
    .result-item {
      cursor: pointer;
      padding: 6px 10px;
      margin-bottom: 4px;
      border-radius: 20px;
      transition: transform 0.1s ease, box-shadow 0.1s ease;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-size: 13px;
      color: white;
      text-align: center;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .result-item:hover {
      transform: translateY(-1px);
      box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .result-item.selected {
      outline: 1px solid var(--border-color, #cccccc);
      transform: scale(1.05);
      box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .content-container {
      display: flex;
      justify-content: space-between;
      align-items: center;
      width: 100%;
    }
    
    .name-container {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    
    .lang-badge {
      font-size: 9px;
      padding: 1px 4px;
      background-color: rgba(255,255,255,0.2);
      border-radius: 3px;
      margin-left: 4px;
      flex-shrink: 0;
    }
  `;

  constructor() {
    super();
    this.results = [];
    this.visible = false;
    this.selectedIndex = -1;
  }

  render() {
    return html`
      <h3>Search Results</h3>
      <div id="results-list">
        ${this.results.map((result, index) => html`
          <div class="result-item ${index === this.selectedIndex ? 'selected' : ''}"
               style="background-color: ${constants.nodeTypeColors[result.node_type] || '#333'}"
               @click="${() => this._selectResult(result)}"
               @mouseenter="${() => this.selectedIndex = index}"
               @mouseover="${() => this._hoverResult(result)}"
               @mouseout="${this._hoverOut}">
            <div class="content-container">
              <div class="name-container">${result.properties.name || 'Unnamed'}</div>
              ${result.properties.file ? html`
                <span class="lang-badge">${fileUtils.fileExtension(result.properties.file)}</span>
              ` : ''}
            </div>
          </div>
        `)}
      </div>
    `;
  }

  updated(changedProps) {
    if (changedProps.has('selectedIndex')) {
      this._scrollToSelected();
    }
  }

  _scrollToSelected() {
    if (this.selectedIndex >= 0) {
      setTimeout(() => {
        const selected = this.shadowRoot.querySelector('.selected');
        if (selected) {
          selected.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
      }, 0);
    }
  }

  _selectResult(result) {
    emit(this, 'select-result', { result });
  }

  _hoverResult(result) {
    emit(this, 'hover-result', { result });
  }

  _hoverOut() {
    emit(this, 'hover-out');
  }

  selectNext() {
    if (this.results.length === 0) return;
    this.selectedIndex = (this.selectedIndex + 1) % this.results.length;
  }

  selectPrevious() {
    if (this.results.length === 0) return;
    this.selectedIndex = (this.selectedIndex - 1 + this.results.length) % this.results.length;
  }

  getSelectedResult() {
    return this.results[this.selectedIndex];
  }
}

customElements.define('tagger-results', TaggerResults);