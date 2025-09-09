import { useEffect, useRef, useCallback } from "https://esm.sh/preact/hooks";
import * as utils from "./utils.js";
const { html } = utils;

export const Editor = ({
  onSearchTriggered,
  onTagClick,
  onHideTooltip,
  onRangeChange,
  onKeyDown,
}) => {
  const editorRef = useRef(null);

  // Initialize search timeout ref
  const searchTimeoutRef = useRef(null);

  useEffect(() => {
    // Focus editor on mount
    if (editorRef.current) {
      editorRef.current.focus();
    }
  }, []);

  const findTriggerBeforeCursor = (range) => {
    if (!range.collapsed) return null;

    const textNode = range.startContainer;
    if (textNode.nodeType !== Node.TEXT_NODE) return null;

    const text = textNode.textContent;
    const cursorPos = range.startOffset;
    // Find the start of the current word or expression
    let wordStart = cursorPos;
    // Go backward until we find a space or beginning of text
    for (let i = cursorPos - 1; i >= 0; i--) {
      // If we find a space, we've reached the beginning of the current word
      if (/\s/.test(text[i])) {
        break;
      }
      wordStart = i;
    }
    // Now check if this word starts with a trigger character
    if (
      wordStart < text.length &&
      (text[wordStart] === "#" ||
        text[wordStart] === "/" ||
        text[wordStart] === "@")
    ) {
      const triggerChar = text[wordStart];
      const tagText = text.substring(wordStart + 1, cursorPos);

      // Make sure the tag contains only valid characters
      return tagText && /^[\w\.-\/]+$/.test(tagText)
        ? { text: tagText, trigger: triggerChar }
        : null;
    }

    return null;
  };

  const cleanupEmptyTaggedSpans = () => {
    const editor = editorRef.current;
    if (!editor) return false;

    // Get the raw HTML content
    const rawContent = editor.innerHTML;
    const textContent = editor.textContent.trim();

    // More aggressive detection of "empty" state
    if (
      textContent === "" ||
      rawContent === "" ||
      rawContent === "<br>" ||
      rawContent === "&nbsp;" ||
      /^(\s|&nbsp;|<br\s*\/?>|<div>(<br\s*\/?>)*<\/div>)*$/i.test(rawContent)
    ) {
      // Complete reset
      editor.innerHTML = "";

      // Reset saved state
      onRangeChange(null);

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
  };

  const handleInput = useCallback(() => {
    // Check if the editor is empty and reset it if needed
    if (cleanupEmptyTaggedSpans()) {
      return; // Exit early if we've cleaned up
    }

    // Clear previous timeout
    clearTimeout(searchTimeoutRef.current);

    // Set new timeout
    searchTimeoutRef.current = setTimeout(() => {
      const selection = window.getSelection();
      if (!selection.rangeCount) return;

      const range = selection.getRangeAt(0);
      const triggerResult = findTriggerBeforeCursor(range);

      if (triggerResult) {
        // Serialize range info to preserve it
        const rangeInfo = {
          text: range.startContainer.textContent,
          startOffset: range.startOffset,
          triggerStart: range.startOffset - triggerResult.text.length - 1,
          triggerChar: triggerResult.trigger,
          tagText: triggerResult.text,
        };

        onRangeChange(rangeInfo);
        onSearchTriggered(triggerResult.text, triggerResult.trigger);
      }
    }, 222);
  }, [onSearchTriggered, onRangeChange]);

  const handleEditorClick = useCallback(
    (e) => {
      // Check if we clicked on a tagged word
      if (e.target.closest(".tagged-text")) {
        const tagElement = e.target.closest(".tagged-text");
        onTagClick(tagElement);
      } else {
        // If we clicked elsewhere in the editor, hide the tooltip
        onHideTooltip();
      }
    },
    [onTagClick, onHideTooltip]
  );

  const handleKeyUp = useCallback((e) => {
    if (e.key === "Backspace" || e.key === "Delete") {
      cleanupEmptyTaggedSpans();
    }
  }, []);

  return html`
    <div id="editor-container">
      <div
        id="editor"
        contenteditable="true"
        ref=${editorRef}
        onInput=${handleInput}
        onClick=${handleEditorClick}
        onKeyDown=${onKeyDown}
        onKeyUp=${handleKeyUp}
      ></div>
    </div>
  `;
};

export const Tooltip = ({
  bodyText,
  docsText,
  filePath,
  isVisible,
  onClose,
  onUntag,
  onMouseEnter,
  onMouseLeave,
}) => {
  const codeRef = useRef(null);
  const docsRef = useRef(null);

  useEffect(() => {
    if (isVisible && bodyText && codeRef.current) {
      codeRef.current.textContent = bodyText;
      hljs.highlightElement(codeRef.current);
    }
  }, [isVisible, bodyText]);

  if (!isVisible) return null;

  const langClass = utils.getHighlightJsClass(filePath);

  return html`
    <div
      class="tooltip"
      style="display: block;"
      onMouseEnter=${onMouseEnter}
      onMouseLeave=${onMouseLeave}
    >
      <div class="tooltip-header">
        <div class="file-info">
          <span class="file-path">
            ${filePath || "Unknown file"}
            ${filePath &&
            html`
              <span class="file-type">
                ${utils.getLanguageFromFilePath(filePath)}
              </span>
            `}
          </span>
          <div class="actions">
            <div class="untag-btn" onClick=${onUntag}>Untag</div>
            <div class="close-btn" onClick=${onClose}>×</div>
          </div>
        </div>
      </div>
      <div class="tooltip-content">
        ${docsText &&
        html`
          <div class="docs-section">
            <div class="docs-header">Documentation</div>
            <div class="docs-content" ref=${docsRef}>${docsText}</div>
          </div>
        `}
        ${bodyText
          ? html`
              <div class="code-section">
                ${docsText && html`<div class="code-header">Source Code</div>`}
                <pre><code class=${langClass} ref=${codeRef}>${bodyText}</code></pre>
              </div>
            `
          : !docsText && "No content available"}
      </div>
    </div>
  `;
};

export const ResultsPane = ({
  results,
  isVisible,
  selectedIndex,
  onResultClick,
  onResultMouseEnter,
  onResultMouseOut,
}) => {
  if (!isVisible) return null;

  return html`
    <div id="results-pane" style="display: block;">
      <h3>Code (${results.length})</h3>
      <div id="results-list">
        ${results.map(
          (result, index) => html`
            <div
              class="result-item ${index === selectedIndex ? "selected" : ""}"
              style="background-color: ${utils.NODE_TYPE_COLORS[
                result.node_type
              ] || "#333"}"
              onClick=${() => onResultClick(result)}
              onMouseEnter=${() => onResultMouseEnter(index, result)}
              onMouseOut=${onResultMouseOut}
              key=${index}
              data-body=${result.properties.body || ""}
              data-file=${result.properties.file || ""}
            >
              <div
                style="display: flex; justify-content: space-between; align-items: center; width: 100%;"
              >
                <div
                  style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"
                >
                  ${result.properties.name || "Unnamed"}
                </div>
                ${result.properties.file &&
                html`
                  <span
                    class="lang-badge"
                    style="font-size: 9px; padding: 1px 4px; background-color: rgba(255,255,255,0.2); border-radius: 3px; margin-left: 4px; flex-shrink: 0;"
                  >
                    ${utils.fileExtension(result.properties.file)}
                  </span>
                `}
              </div>
            </div>
          `
        )}
      </div>
    </div>
  `;
};
