import { LitElement, html, css } from 'https://cdn.jsdelivr.net/gh/lit/dist@2/core/lit-core.min.js';

class TaggerApp extends LitElement {
  static properties = {
    baseUrl: { type: String, attribute: 'base-url' },
    darkMode: { type: Boolean, reflect: true },
    shiftPressed: { type: Boolean, state: true }
  };

  static styles = css`
    :host {
      display: block;
      color: var(--text-color);
      background-color: var(--bg-color);
    }
    
    .content-container {
      display: flex;
      position: relative;
    }
  `;

  constructor() {
    super();
    this.baseUrl = '';
    this.darkMode = false;
    this.shiftPressed = false;
    
    // Check for dark mode
    this._checkDarkMode();
    
    // Listen for system dark mode changes
    if (window.matchMedia) {
      window.matchMedia('(prefers-color-scheme: dark)')
        .addEventListener('change', () => this._checkDarkMode());
    }
  }

  render() {
    return html`
      <div class="content-container">
        <tagger-tooltip id="tooltip" @untag="${this._handleUntag}"></tagger-tooltip>
        
        <tagger-editor id="editor"
          @tag-input="${this._handleTagInput}"
          @tag-input-end="${this._hideResults}"
          @escape-key="${this._handleEscape}"
          @navigate-down="${() => this._results.selectNext()}"
          @navigate-up="${() => this._results.selectPrevious()}"
          @enter-key="${this._handleEnterKey}"
          @shift-key="${this._handleShiftKey}"
          @tag-click="${this._handleTagClick}"
          @editor-click="${() => this._tooltip.hide()}">
        </tagger-editor>
        
        <tagger-results id="results"
          @select-result="${this._handleSelectResult}"
          @hover-result="${this._handleHoverResult}"
          @hover-out="${this._handleHoverOut}">
        </tagger-results>
      </div>
    `;
  }

  get _editor() {
    return this.shadowRoot.querySelector('#editor');
  }
  
  get _tooltip() {
    return this.shadowRoot.querySelector('#tooltip');
  }
  
  get _results() {
    return this.shadowRoot.querySelector('#results');
  }

  _checkDarkMode() {
    this.darkMode = window.matchMedia && 
      window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  _handleTagInput(e) {
    const { tag, trigger } = e.detail;
    this._searchForTag(tag, trigger);
  }

  _hideResults() {
    this._results.visible = false;
  }

  _handleEscape() {
    this._hideResults();
    this._tooltip.hide();
  }

  _handleEnterKey() {
    const result = this._results.getSelectedResult();
    if (result) {
      // Instead of directly calling replaceTagWithStyledSpan, use the working method
      this._handleSelectResult({ detail: { result } });
    }
  }

  _handleShiftKey(e) {
    this.shiftPressed = e.detail.pressed;
    
    if (!this.shiftPressed) {
      // Hide tooltip when shift is released
      this._tooltip.hide();
    } else if (this._results.visible) {
      // Show tooltip for selected result when shift is pressed
      const result = this._results.getSelectedResult();
      if (result) {
        this._showTooltip(result.properties.body, result.properties.file);
      }
    }
  }

  _handleTagClick(e) {
    this._showTooltip(e.detail.body, e.detail.file);
  }

  _handleSelectResult(e) {
    const { result } = e.detail;
    this._tooltip.hide(); // Add this line to hide tooltip before replacing tag
    this._editor.replaceTagWithStyledSpan(result);
    this._hideResults();
  }

  _handleHoverResult(e) {
    if (!this.shiftPressed) {
      const { result } = e.detail;
      this._showTooltip(result.properties.body, result.properties.file);
    }
  }

  _handleHoverOut() {
    if (!this.shiftPressed) {
      this._tooltip.hide();
    }
  }

  _handleUntag() {
    this._editor.untagActiveElement();
  }

  _showTooltip(body, file) {
    this._tooltip.show(body, file);
  }

  async _searchForTag(tag, trigger) {
    try {
      let queryParams = `query=${encodeURIComponent(tag)}`;
      
      // Add node_type param based on trigger
      if (trigger === '/') {
        queryParams += '&node_types=Page';
      } else if (trigger === '@') {
        queryParams += '&node_types=File';
      }
      
      const response = await fetch(`${this.baseUrl}/search?${queryParams}`);
      if (!response.ok) throw new Error(`HTTP error ${response.status}`);
      
      const data = await response.json();
      
      // Update and show results if we have any
      this._results.results = data;
      if (data && data.length > 0) {
        this._results.visible = true;
        this._results.selectedIndex = 0;
      } else {
        this._hideResults();
      }
    } catch (error) {
      console.error("Search error:", error);
      this._hideResults();
    }
  }
}

customElements.define('tagger-app', TaggerApp);