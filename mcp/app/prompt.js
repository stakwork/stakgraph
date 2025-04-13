import { h } from 'https://esm.sh/preact';
import { useState, useEffect, useRef, useCallback } from 'https://esm.sh/preact/hooks';
import htm from 'https://esm.sh/htm';

// Initialize HTM with Preact
const html = htm.bind(h);

// Constants
const NODE_TYPE_COLORS = {
    "Repository": "#2C5985", // Darker blue
    "Language": "#A35D00",   // Darker orange
    "Directory": "#3A7336",  // Darker green
    "File": "#ad8cc6",       // Darker purple
    "Import": "#8B2E2A",     // Darker red
    "Class": "#4A7D4A",      // Darker light green
    "Trait": "#3B6EB5",      // Darker light blue
    "Library": "#A83333",    // Darker pink
    "Function": "#C67000",   // Darker light orange
    "Test": "#B7940A",       // Darker yellow
    "E2etest": "#7C4A85",    // Darker lavender
    "Endpoint": "#385D8A",   // Darker blue gray
    "Request": "#6B4A7A",    // Darker medium purple
    "Datamodel": "#A13939",  // Darker salmon
    "Page": "#2980B9"        // Darker sky blue
};

// Utility functions
const fileExtension = (filePath) => {
    if (!filePath) return null;
    return filePath.split('.').pop().toLowerCase();
};

const getLanguageFromFilePath = (filePath) => {
    if (!filePath) return null;
    const extension = filePath.split('.').pop().toLowerCase();
    const langMap = {
        'js': 'JavaScript',
        'jsx': 'React',
        'ts': 'TypeScript',
        'tsx': 'React TSX',
        'py': 'Python',
        'rb': 'Ruby',
        'java': 'Java',
        'go': 'Go',
        'rs': 'Rust',
        'c': 'C',
        'cpp': 'C++',
        'cs': 'C#',
        'php': 'PHP',
        'html': 'HTML',
        'css': 'CSS',
        'scss': 'SCSS',
        'json': 'JSON',
        'md': 'Markdown',
        'sql': 'SQL',
        'sh': 'Shell',
        'bash': 'Bash',
        'yaml': 'YAML',
        'yml': 'YAML',
        'xml': 'XML'
    };
    
    return langMap[extension] || extension.toUpperCase();
};

const getHighlightJsClass = (filePath) => {
    if (!filePath) return '';
    
    const extension = filePath.split('.').pop().toLowerCase();
    const langMap = {
        'js': 'language-javascript',
        'jsx': 'language-javascript',
        'ts': 'language-typescript',
        'tsx': 'language-typescript',
        'py': 'language-python',
        'rb': 'language-ruby',
        'java': 'language-java',
        'go': 'language-go',
        'rs': 'language-rust',
        'c': 'language-c',
        'cpp': 'language-cpp',
        'cs': 'language-csharp',
        'php': 'language-php',
        'html': 'language-html',
        'css': 'language-css',
        'scss': 'language-scss',
        'json': 'language-json',
        'md': 'language-markdown',
        'sql': 'language-sql',
        'sh': 'language-bash',
        'bash': 'language-bash',
        'yaml': 'language-yaml',
        'yml': 'language-yaml',
        'xml': 'language-xml'
    };
    
    return langMap[extension] || '';
};

// Sub-components
const Tooltip = ({ bodyText, filePath, isVisible, onClose, onUntag, onMouseEnter, onMouseLeave }) => {
    const codeRef = useRef(null);
    
    useEffect(() => {
        if (isVisible && bodyText && codeRef.current) {
            codeRef.current.textContent = bodyText;
            hljs.highlightElement(codeRef.current);
        }
    }, [isVisible, bodyText]);
    
    if (!isVisible) return null;
    
    const langClass = getHighlightJsClass(filePath);
    
    return html`
        <div class="tooltip" 
            style="display: block;" 
            onMouseEnter=${onMouseEnter}
            onMouseLeave=${onMouseLeave}
        >
            <div class="file-info">
                <span class="file-path">
                    ${filePath || 'Unknown file'}
                    ${filePath && html`
                        <span class="file-type">
                            ${getLanguageFromFilePath(filePath)}
                        </span>
                    `}
                </span>
                <div class="actions">
                    <div class="untag-btn" onClick=${onUntag}>Untag</div>
                    <div class="close-btn" onClick=${onClose}>×</div>
                </div>
            </div>
            ${bodyText 
                ? html`<pre><code class=${langClass} ref=${codeRef}>${bodyText}</code></pre>` 
                : 'No content available'
            }
        </div>
    `;
};

const ResultsPane = ({ 
    results, 
    isVisible, 
    selectedIndex, 
    onResultClick, 
    onResultMouseEnter,
    onResultMouseOut
}) => {
    if (!isVisible) return null;
    
    return html`
        <div id="results-pane" style="display: block;">
            <h3>Code (${results.length})</h3>
            <div id="results-list">
                ${results.map((result, index) => html`
                    <div 
                        class="result-item ${index === selectedIndex ? 'selected' : ''}"
                        style="background-color: ${NODE_TYPE_COLORS[result.node_type] || '#333'}"
                        onClick=${() => onResultClick(result)}
                        onMouseEnter=${() => onResultMouseEnter(index, result)}
                        onMouseOut=${onResultMouseOut}
                        key=${index}
                        data-body=${result.properties.body || ''}
                        data-file=${result.properties.file || ''}
                    >
                        <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
                            <div style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                                ${result.properties.name || 'Unnamed'}
                            </div>
                            ${result.properties.file && html`
                                <span class="lang-badge"
                                    style="font-size: 9px; padding: 1px 4px; background-color: rgba(255,255,255,0.2); border-radius: 3px; margin-left: 4px; flex-shrink: 0;"
                                >
                                    ${fileExtension(result.properties.file)}
                                </span>
                            `}
                        </div>
                    </div>
                `)}
            </div>
        </div>
    `;
};

const Editor = ({ 
    onSearchTriggered, 
    onTagClick, 
    onHideTooltip,
    lastRange, 
    onRangeChange,
    currentTag,
    setCurrentTag,
    currentTrigger,
    setCurrentTrigger,
    onKeyDown
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
        
        // Find the start of the triggered text
        let triggerStart = -1;
        let triggerChar = '';
        
        // Check for #, /, or @ triggers
        for (let i = cursorPos - 1; i >= 0; i--) {
            if (text[i] === '#' || text[i] === '/' || text[i] === '@') {
                triggerStart = i;
                triggerChar = text[i];
                break;
            } else if (!/[\w\.-]/.test(text[i])) { // Allow . and -
                break;
            }
        }
        
        if (triggerStart === -1) return null;
        
        // Extract the text without the trigger symbol
        const tagText = text.substring(triggerStart + 1, cursorPos);
        
        // Return if valid . and -
        return tagText && /^[\w\.-]+$/.test(tagText) ? { text: tagText, trigger: triggerChar } : null;
    };
    
    const cleanupEmptyTaggedSpans = () => {
        const editor = editorRef.current;
        if (!editor) return false;
        
        // Get the raw HTML content
        const rawContent = editor.innerHTML;
        const textContent = editor.textContent.trim();
        
        // More aggressive detection of "empty" state
        if (textContent === '' || 
            rawContent === '' || 
            rawContent === '<br>' || 
            rawContent === '&nbsp;' || 
            /^(\s|&nbsp;|<br\s*\/?>|<div>(<br\s*\/?>)*<\/div>)*$/i.test(rawContent)) {
                            
            // Complete reset
            editor.innerHTML = '';
            
            // Reset saved state
            setCurrentTag('');
            setCurrentTrigger('');
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
                setCurrentTag(triggerResult.text);
                setCurrentTrigger(triggerResult.trigger);
                
                // Serialize range info to preserve it
                const rangeInfo = {
                    text: range.startContainer.textContent,
                    startOffset: range.startOffset,
                    triggerStart: range.startOffset - triggerResult.text.length - 1,
                    triggerChar: triggerResult.trigger,
                    tagText: triggerResult.text
                };
                
                onRangeChange(rangeInfo);
                onSearchTriggered(triggerResult.text, triggerResult.trigger);
            }
        }, 222);
    }, [onSearchTriggered, setCurrentTag, setCurrentTrigger, onRangeChange]);
    
    const handleEditorClick = useCallback((e) => {
        // Check if we clicked on a tagged word
        if (e.target.closest('.tagged-text')) {
            const tagElement = e.target.closest('.tagged-text');
            onTagClick(tagElement);
        } else {
            // If we clicked elsewhere in the editor, hide the tooltip
            onHideTooltip();
        }
    }, [onTagClick, onHideTooltip]);
    
    const handleKeyUp = useCallback((e) => {
        if (e.key === 'Backspace' || e.key === 'Delete') {
            cleanupEmptyTaggedSpans();
        }
    }, []);
    
    return html`
        <div id="editor-container">
            <div 
                id="editor" 
                contentEditable="true"
                ref=${editorRef}
                onInput=${handleInput}
                onClick=${handleEditorClick}
                onKeyDown=${onKeyDown}
                onKeyUp=${handleKeyUp}
            ></div>
        </div>
    `;
};

// Main Prompt Component
export const Prompt = ({ onSend }) => {
    // State
    const [darkMode, setDarkMode] = useState(
        window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
    );
    const [tooltipVisible, setTooltipVisible] = useState(false);
    const [tooltipBodyText, setTooltipBodyText] = useState('');
    const [tooltipFilePath, setTooltipFilePath] = useState('');
    const [results, setResults] = useState([]);
    const [showResults, setShowResults] = useState(false);
    const [selectedResultIndex, setSelectedResultIndex] = useState(-1);
    const [shiftPressed, setShiftPressed] = useState(false);
    const [currentTag, setCurrentTag] = useState('');
    const [currentTrigger, setCurrentTrigger] = useState('');
    const [rangeInfo, setRangeInfo] = useState(null);
    const [activeTagElement, setActiveTagElement] = useState(null);
    const [isHoveringResult, setIsHoveringResult] = useState(false);
    
    // Refs
    const tooltipTimeoutRef = useRef(null);
    const editorRef = useRef(null);
    
    // Set editor ref when mounted
    useEffect(() => {
        editorRef.current = document.getElementById('editor');
    }, []);
    
    // Config
    const baseUrl = window.BASE_URL || "";
    
    // Check for dark mode
    useEffect(() => {
        const darkModeMediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
        const handleChange = (e) => setDarkMode(e.matches);
        
        if (darkMode) {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
        
        darkModeMediaQuery.addEventListener('change', handleChange);
        return () => darkModeMediaQuery.removeEventListener('change', handleChange);
    }, [darkMode]);
    
    // Global keydown/keyup for shift
    useEffect(() => {
        const handleGlobalKeydown = (e) => {
            if (e.key === 'Shift') {
                setShiftPressed(true);
                if (showResults && selectedResultIndex >= 0 && selectedResultIndex < results.length) {
                    const selectedItem = results[selectedResultIndex];
                    showTooltip(
                        selectedItem.properties.body, 
                        selectedItem.properties.file
                    );
                }
            }
        };
        
        const handleGlobalKeyup = (e) => {
            if (e.key === 'Shift') {
                setShiftPressed(false);
                // Hide tooltip on shift release unless actively hovering something
                if (!isHoveringResult) {
                    hideTooltip();
                }
            }
        };
        
        document.addEventListener('keydown', handleGlobalKeydown);
        document.addEventListener('keyup', handleGlobalKeyup);
        
        return () => {
            document.removeEventListener('keydown', handleGlobalKeydown);
            document.removeEventListener('keyup', handleGlobalKeyup);
        };
    }, [showResults, selectedResultIndex, results, isHoveringResult]);
    
    // Global click handler
    useEffect(() => {
        const handleGlobalClick = (e) => {
            if (!e.target.closest('.tooltip') && 
                !e.target.closest('#editor') && 
                !e.target.closest('.result-item')) {
                hideTooltip();
                hideResultsPane();
                setActiveTagElement(null);
            }
        };
        
        document.addEventListener('click', handleGlobalClick);
        return () => document.removeEventListener('click', handleGlobalClick);
    }, []);

    useEffect(() => {
        if (showResults && selectedResultIndex >= 0 && results.length > 0) {
            // Find the selected element and scroll it into view
            const selectedElement = document.querySelector('.result-item.selected');
            if (selectedElement) {
                selectedElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'nearest'
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
        setTooltipBodyText(bodyText || '');
        setTooltipFilePath(filePath || '');
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
            if (trigger === '/') {
                queryParams += '&node_types=Page,Endpoint';
            } else if (trigger === '@') {
                queryParams += '&node_types=File';
            }
            
            const response = await fetch(`${baseUrl}/search?${queryParams}`);
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

    const replaceTagWithStyledSpan = useCallback((result, currentRangeInfo) => {
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
            const triggerIndex = text.lastIndexOf(currentRangeInfo.triggerChar + currentRangeInfo.tagText);
            
            if (triggerIndex !== -1) {
                // Verify this is the most recent occurrence (closest to cursor)
                foundNode = node;
                triggerPos = triggerIndex;
                
                // We found a match - now confirm it's at the current cursor position
                // by checking if the position after the tag matches our saved range
                const afterTagPos = triggerIndex + currentRangeInfo.tagText.length + 1;
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
        const color = NODE_TYPE_COLORS[result.node_type] || '#333';
        const tagSpan = document.createElement('span');
        tagSpan.className = 'tagged-text';
        tagSpan.style.color = color;
        tagSpan.style.fontWeight = 'bold';
        
        // For Files, show the filename
        if (result.node_type === 'File' && result.properties.name) {
            tagSpan.textContent = `@${result.properties.name}`;
        } else {
            tagSpan.textContent = `${currentRangeInfo.triggerChar}${currentRangeInfo.tagText}`;
        }
        tagSpan.dataset.body = result.properties.body || '';
        tagSpan.dataset.file = result.properties.file || '';
        tagSpan.dataset.nodeType = result.node_type;
        
        // Add styled span to fragment
        fragment.appendChild(tagSpan);
        
        // Create a separate text node for afterText with zero-width space if empty
        const afterNode = document.createTextNode(afterText || '\u200B');
        fragment.appendChild(afterNode);
        
        // Replace the original text node with our new structure
        foundNode.parentNode.replaceChild(fragment, foundNode);
        
        // Position cursor after the tagged span
        const range = document.createRange();
        if (afterText === '') {
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
        setCurrentTag('');
        setCurrentTrigger('');
        setRangeInfo(null);
        
        // Ensure editor is focused after insertion
        if (editorRef.current) {
            editorRef.current.focus();
        }
    }, [editorRef]);
    
    const hideResultsPane = useCallback(() => {
        setShowResults(false);
        setSelectedResultIndex(-1);
    }, []);
    
    const handleResultClick = useCallback((result) => {
        replaceTagWithStyledSpan(result, rangeInfo);
        hideResultsPane();
    }, [replaceTagWithStyledSpan, rangeInfo, hideResultsPane]);
    
    const handleResultMouseEnter = useCallback((index, result) => {
        // Clear any existing hide tooltip timeout
        if (tooltipTimeoutRef.current) {
            clearTimeout(tooltipTimeoutRef.current);
            tooltipTimeoutRef.current = null;
        }
        
        setSelectedResultIndex(index);
        setIsHoveringResult(true);
        // Always show tooltip on hover, regardless of shift key
        showTooltip(result.properties.body, result.properties.file);
    }, [showTooltip]);

    const handleResultMouseOut = useCallback(() => {
        // Use a small delay to let other mouseEnter events fire first
        setTimeout(() => {
            // Only hide if we're not hovering any result or the tooltip
            const isHoveringAnyResult = document.querySelector('.result-item:hover');
            const isHoveringTooltip = document.querySelector('.tooltip:hover');
            
            if (!isHoveringAnyResult && !isHoveringTooltip && !shiftPressed) {
                setIsHoveringResult(false);
                hideTooltip();
            }
        }, 50);
    }, [shiftPressed, hideTooltip]);
    
    const handleTagClick = useCallback((tagElement) => {
        // If clicking on the same tag that's already active, do nothing
        if (tagElement === activeTagElement && tooltipVisible) {
            return;
        }
        
        // Set this as the active tag
        setActiveTagElement(tagElement);
        
        // Show the tooltip for this tag
        showTooltip(
            tagElement.dataset.body, 
            tagElement.dataset.file
        );
    }, [activeTagElement, tooltipVisible, showTooltip]);
    
    // Collect tagged words from the editor
    const collectTaggedWords = useCallback(() => {
        if (!editorRef.current) return [];
        
        const taggedElements = editorRef.current.querySelectorAll('.tagged-text');
        return Array.from(taggedElements).map(el => ({
            text: el.textContent,
            nodeType: el.dataset.nodeType || null,
            file: el.dataset.file || null,
            body: el.dataset.body || null
        }));
    }, []);
    
    const handleKeyDown = useCallback((e) => {
        // Close results pane and tooltip on escape
        if (e.key === 'Escape') {
            hideResultsPane();
            hideTooltip();
            return;
        }
        
        // Handle Enter key - emit the send event
        if (e.key === 'Enter' && !e.shiftKey) {
            if (!showResults) { // Only if results pane is not showing
                e.preventDefault();
                if (editorRef.current) {
                    const text = editorRef.current.textContent || '';
                    const taggedWords = collectTaggedWords();
                    if (onSend && text.trim()) {
                        onSend({
                            text,
                            taggedWords
                        });
                    }
                }
            }
        }
        
        // Handle keyboard navigation in search results
        if (showResults && results.length > 0) {
            if (e.key === 'ArrowDown') {
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
            } else if (e.key === 'ArrowUp') {
                e.preventDefault(); // Prevent cursor movement
                const newIndex = (selectedResultIndex - 1 + results.length) % results.length;
                setSelectedResultIndex(newIndex);
                
                // Use newIndex directly
                if (shiftPressed && results[newIndex]) {
                    const selectedItem = results[newIndex];
                    showTooltip(
                        selectedItem.properties.body,
                        selectedItem.properties.file
                    );
                }
            } else if (e.key === 'Enter' && selectedResultIndex >= 0) {
                e.preventDefault(); // Prevent default Enter behavior
                const selectedResult = results[selectedResultIndex];
                handleResultClick(selectedResult);
            }
        }
    }, [
        showResults, 
        results, 
        selectedResultIndex, 
        shiftPressed, 
        showTooltip, 
        hideResultsPane, 
        hideTooltip, 
        handleResultClick, 
        onSend, 
        collectTaggedWords
    ]);
    
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
                lastRange=${rangeInfo}
                onRangeChange=${setRangeInfo}
                currentTag=${currentTag}
                setCurrentTag=${setCurrentTag}
                currentTrigger=${currentTrigger}
                setCurrentTrigger=${setCurrentTrigger}
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