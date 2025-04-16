import {
  useState,
  useEffect,
  useRef,
  useCallback,
} from "https://esm.sh/preact/hooks";
import { Editor, Tooltip, ResultsPane } from "./editor.js";
import { NODE_TYPE_COLORS, html } from "./utils.js";

export const Prompt = ({ onSend }) => {
  const [tooltipVisible, setTooltipVisible] = useState(false);
  const [tooltipBodyText, setTooltipBodyText] = useState("");
  const [tooltipFilePath, setTooltipFilePath] = useState("");
  const [results, setResults] = useState([]);
  const [showResults, setShowResults] = useState(false);
  const [selectedResultIndex, setSelectedResultIndex] = useState(-1);
  const [shiftPressed, setShiftPressed] = useState(false);
  const [rangeInfo, setRangeInfo] = useState(null);
  const [activeTagElement, setActiveTagElement] = useState(null);
  const [isHoveringResult, setIsHoveringResult] = useState(false);

  // Refs
  const tooltipTimeoutRef = useRef(null);
  const editorRef = useRef(null);

  // Set editor ref when mounted
  useEffect(() => {
    editorRef.current = document.getElementById("editor");
  }, []);

  // Config
  const baseUrl = window.BASE_URL || "";

  // Global keydown/keyup for shift
  useEffect(() => {
    const handleGlobalKeydown = (e) => {
      if (e.key === "Shift") {
        setShiftPressed(true);
        if (
          showResults &&
          selectedResultIndex >= 0 &&
          selectedResultIndex < results.length
        ) {
          const selectedItem = results[selectedResultIndex];
          showTooltip(
            selectedItem.properties.body,
            selectedItem.properties.file
          );
        }
      }
    };

    const handleGlobalKeyup = (e) => {
      if (e.key === "Shift") {
        setShiftPressed(false);
        // Hide tooltip on shift release unless actively hovering something
        if (!isHoveringResult) {
          hideTooltip();
        }
      }
    };

    document.addEventListener("keydown", handleGlobalKeydown);
    document.addEventListener("keyup", handleGlobalKeyup);

    return () => {
      document.removeEventListener("keydown", handleGlobalKeydown);
      document.removeEventListener("keyup", handleGlobalKeyup);
    };
  }, [showResults, selectedResultIndex, results, isHoveringResult]);

  // Global click handler
  useEffect(() => {
    const handleGlobalClick = (e) => {
      if (
        !e.target.closest(".tooltip") &&
        !e.target.closest("#editor") &&
        !e.target.closest(".result-item")
      ) {
        hideTooltip();
        hideResultsPane();
        setActiveTagElement(null);
      }
    };

    document.addEventListener("click", handleGlobalClick);
    return () => document.removeEventListener("click", handleGlobalClick);
  }, []);

  useEffect(() => {
    if (showResults && selectedResultIndex >= 0 && results.length > 0) {
      // Find the selected element and scroll it into view
      const selectedElement = document.querySelector(".result-item.selected");
      if (selectedElement) {
        selectedElement.scrollIntoView({
          behavior: "smooth",
          block: "nearest",
        });
      }
    }
  }, [selectedResultIndex, showResults, results]);

  // Tooltip methods
  const showTooltip = useCallback((bodyText, filePath) => {
    if (tooltipTimeoutRef.current) {
      clearTimeout(tooltipTimeoutRef.current);
      tooltipTimeoutRef.current = null;
    }
    setTooltipBodyText(bodyText || "");
    setTooltipFilePath(filePath || "");
    setTooltipVisible(true);
  }, []);

  const hideTooltip = useCallback(() => {
    setTooltipVisible(false);
  }, []);

  const untagElement = useCallback(() => {
    if (!activeTagElement) return;

    // Get the text content
    const tagText = activeTagElement.textContent;

    // Create a text node to replace the tag span
    const textNode = document.createTextNode(tagText);

    // Replace the tag span with the text node
    activeTagElement.parentNode.replaceChild(textNode, activeTagElement);

    // Reset active tag
    setActiveTagElement(null);
    hideTooltip();
  }, [activeTagElement, hideTooltip]);

  // Search and results methods
  const searchForTag = useCallback(async (tag, trigger) => {
    try {
      let queryParams = `query=${encodeURIComponent(tag)}`;

      // Add node_type param based on trigger
      if (trigger === "/") {
        queryParams += "&node_types=Page,Endpoint";
      } else if (trigger === "@") {
        queryParams += "&node_types=File";
      }

      const response = await fetch(`${baseUrl}/search?${queryParams}&limit=40`);
      if (!response.ok) throw new Error(`HTTP error ${response.status}`);

      const data = await response.json();
      setResults(data);
      setShowResults(data && data.length > 0);
      setSelectedResultIndex(data && data.length > 0 ? 0 : -1);
    } catch (error) {
      console.error("Search error:", error);
      hideResultsPane();
    }
  }, []);

  const replaceTagWithStyledSpan = useCallback(
    (result, currentRangeInfo) => {
      if (!currentRangeInfo || !editorRef.current) {
        console.error("No range information available for tag insertion");
        return;
      }

      // Create a fresh selection and range based on the saved range info
      const selection = window.getSelection();
      selection.removeAllRanges();

      // Find the text node at current cursor position
      const textNodes = [];
      const collectTextNodes = (node) => {
        if (node.nodeType === Node.TEXT_NODE) {
          textNodes.push(node);
        } else {
          for (let i = 0; i < node.childNodes.length; i++) {
            collectTextNodes(node.childNodes[i]);
          }
        }
      };

      collectTextNodes(editorRef.current);

      // Find the specific text node where our cursor is
      let foundNode = null;
      let triggerPos = -1;

      for (const node of textNodes) {
        const text = node.textContent;
        const triggerIndex = text.lastIndexOf(
          currentRangeInfo.triggerChar + currentRangeInfo.tagText
        );

        if (triggerIndex !== -1) {
          // Verify this is the most recent occurrence (closest to cursor)
          foundNode = node;
          triggerPos = triggerIndex;

          // We found a match - now confirm it's at the current cursor position
          // by checking if the position after the tag matches our saved range
          const afterTagPos =
            triggerIndex + currentRangeInfo.tagText.length + 1;
          if (afterTagPos === currentRangeInfo.startOffset) {
            break; // This is definitely our current tag
          }
        }
      }

      if (!foundNode) {
        console.error("Could not find the text node containing the tag");
        return;
      }

      const text = foundNode.textContent;

      // Split text into before, tag, and after
      const beforeText = text.substring(0, triggerPos);
      const afterPos = triggerPos + currentRangeInfo.tagText.length + 1;
      const afterText = text.substring(afterPos);

      // Create document fragment
      const fragment = document.createDocumentFragment();

      // Add text before the tag
      if (beforeText) {
        fragment.appendChild(document.createTextNode(beforeText));
      }

      // Create styled span for the tag
      const color = NODE_TYPE_COLORS[result.node_type] || "#333";
      const tagSpan = document.createElement("span");
      tagSpan.className = "tagged-text";
      tagSpan.style.color = color;
      tagSpan.style.fontWeight = "bold";

      // For Files, show the filename
      if (result.node_type === "File" && result.properties.name) {
        tagSpan.textContent = `@${result.properties.name}`;
      } else {
        tagSpan.textContent = `${currentRangeInfo.triggerChar}${currentRangeInfo.tagText}`;
      }
      tagSpan.dataset.body = result.properties.body || "";
      tagSpan.dataset.file = result.properties.file || "";
      tagSpan.dataset.nodeType = result.node_type;

      // Add styled span to fragment
      fragment.appendChild(tagSpan);

      // Create a separate text node for afterText with zero-width space if empty
      const afterNode = document.createTextNode(afterText || "\u200B");
      fragment.appendChild(afterNode);

      // Replace the original text node with our new structure
      foundNode.parentNode.replaceChild(fragment, foundNode);

      // Position cursor after the tagged span
      const range = document.createRange();
      if (afterText === "") {
        // If using zero-width space, position after it
        range.setStart(afterNode, 1);
      } else {
        // Otherwise position at beginning of afterText
        range.setStart(afterNode, 0);
      }
      range.collapse(true);
      selection.removeAllRanges();
      selection.addRange(range);

      // Reset state
      setRangeInfo(null);

      // Ensure editor is focused after insertion
      if (editorRef.current) {
        editorRef.current.focus();
      }
    },
    [editorRef]
  );

  const hideResultsPane = useCallback(() => {
    setShowResults(false);
    setSelectedResultIndex(-1);
  }, []);

  const handleResultClick = useCallback(
    (result) => {
      replaceTagWithStyledSpan(result, rangeInfo);
      hideResultsPane();
    },
    [replaceTagWithStyledSpan, rangeInfo, hideResultsPane]
  );

  const handleResultMouseEnter = useCallback(
    (index, result) => {
      // Clear any existing hide tooltip timeout
      if (tooltipTimeoutRef.current) {
        clearTimeout(tooltipTimeoutRef.current);
        tooltipTimeoutRef.current = null;
      }

      setSelectedResultIndex(index);
      setIsHoveringResult(true);
      // Always show tooltip on hover, regardless of shift key
      showTooltip(result.properties.body, result.properties.file);
    },
    [showTooltip]
  );

  const handleResultMouseOut = useCallback(() => {
    // Use a small delay to let other mouseEnter events fire first
    setTimeout(() => {
      // Only hide if we're not hovering any result or the tooltip
      const isHoveringAnyResult = document.querySelector(".result-item:hover");
      const isHoveringTooltip = document.querySelector(".tooltip:hover");

      if (!isHoveringAnyResult && !isHoveringTooltip && !shiftPressed) {
        setIsHoveringResult(false);
        hideTooltip();
      }
    }, 50);
  }, [shiftPressed, hideTooltip]);

  const handleTagClick = useCallback(
    (tagElement) => {
      // If clicking on the same tag that's already active, do nothing
      if (tagElement === activeTagElement && tooltipVisible) {
        return;
      }

      // Set this as the active tag
      setActiveTagElement(tagElement);

      // Show the tooltip for this tag
      showTooltip(tagElement.dataset.body, tagElement.dataset.file);
    },
    [activeTagElement, tooltipVisible, showTooltip]
  );

  // Collect tagged words from the editor
  const collectTaggedWords = useCallback(() => {
    if (!editorRef.current) return [];

    const taggedElements = editorRef.current.querySelectorAll(".tagged-text");
    return Array.from(taggedElements).map((el) => ({
      text: el.textContent,
      nodeType: el.dataset.nodeType || null,
      file: el.dataset.file || null,
      body: el.dataset.body || null,
    }));
  }, []);

  const handleKeyDown = useCallback(
    (e) => {
      // Close results pane and tooltip on escape
      if (e.key === "Escape") {
        hideResultsPane();
        hideTooltip();
        return;
      }

      // Handle Enter key - emit the send event
      if (e.key === "Enter" && !e.shiftKey) {
        if (!showResults) {
          // Only if results pane is not showing
          e.preventDefault();
          if (editorRef.current) {
            const text = editorRef.current.textContent || "";
            const taggedWords = collectTaggedWords();
            if (onSend && text.trim()) {
              onSend({
                text,
                taggedWords,
              });
            }
          }
        }
      }

      // Handle keyboard navigation in search results
      if (showResults && results.length > 0) {
        if (e.key === "ArrowDown") {
          e.preventDefault(); // Prevent cursor movement
          const newIndex = (selectedResultIndex + 1) % results.length;
          setSelectedResultIndex(newIndex);

          // Use newIndex directly
          if (shiftPressed && results[newIndex]) {
            const selectedItem = results[newIndex];
            showTooltip(
              selectedItem.properties.body,
              selectedItem.properties.file
            );
          }
        } else if (e.key === "ArrowUp") {
          e.preventDefault(); // Prevent cursor movement
          const newIndex =
            (selectedResultIndex - 1 + results.length) % results.length;
          setSelectedResultIndex(newIndex);

          // Use newIndex directly
          if (shiftPressed && results[newIndex]) {
            const selectedItem = results[newIndex];
            showTooltip(
              selectedItem.properties.body,
              selectedItem.properties.file
            );
          }
        } else if (e.key === "Enter" && selectedResultIndex >= 0) {
          e.preventDefault(); // Prevent default Enter behavior
          const selectedResult = results[selectedResultIndex];
          handleResultClick(selectedResult);
        }
      }
    },
    [
      showResults,
      results,
      selectedResultIndex,
      shiftPressed,
      showTooltip,
      hideResultsPane,
      hideTooltip,
      handleResultClick,
      onSend,
      collectTaggedWords,
    ]
  );

  return html`
    <div class="content-container">
      <${Tooltip}
        bodyText=${tooltipBodyText}
        filePath=${tooltipFilePath}
        isVisible=${tooltipVisible}
        onClose=${hideTooltip}
        onUntag=${untagElement}
      />
      <${Editor}
        onSearchTriggered=${searchForTag}
        onTagClick=${handleTagClick}
        onHideTooltip=${hideTooltip}
        onRangeChange=${setRangeInfo}
        onKeyDown=${handleKeyDown}
      />
      <${ResultsPane}
        results=${results}
        isVisible=${showResults}
        selectedIndex=${selectedResultIndex}
        onResultClick=${handleResultClick}
        onResultMouseEnter=${handleResultMouseEnter}
        onResultMouseOut=${handleResultMouseOut}
      />
    </div>
  `;
};
