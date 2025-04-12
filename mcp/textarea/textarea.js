// Text area module - handles editor and search functionality
Tagger.Textarea = (function() {
    let searchTimeout;
    return {

    init: function() {
        const editor = Tagger.elements.editor;
        
        // Set up event listeners
        editor.addEventListener('input', this.handleInput.bind(this));
        editor.addEventListener('keydown', this.handleKeydown.bind(this));
        editor.addEventListener('click', this.handleEditorClick.bind(this));
        editor.addEventListener('keyup', this.handleKeyup.bind(this));
        
        // Global keydown/keyup listeners for Shift
        document.addEventListener('keydown', this.handleGlobalKeydown.bind(this));
        document.addEventListener('keyup', this.handleGlobalKeyup.bind(this));
        
        // Global click listener
        document.addEventListener('click', this.handleGlobalClick.bind(this));
    },
    
    handleInput: function(e) {
        // Check if the editor is empty and reset it if needed
        if (this.cleanupEmptyTaggedSpans()) {
            return; // Exit early if we've cleaned up
        }
        
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            const selection = window.getSelection();
            if (!selection.rangeCount) return;
            
            const range = selection.getRangeAt(0);
            const triggerResult = this.findTriggerBeforeCursor(range);
            
            if (triggerResult) {
                Tagger.state.currentTag = triggerResult.text;
                Tagger.state.currentTrigger = triggerResult.trigger;
                Tagger.state.lastRange = range.cloneRange(); // Save the range for later use
                this.searchForTag(triggerResult.text, triggerResult.trigger);
            } else {
                this.hideResultsPane();
            }
        }, 222);
    },
    
    handleKeydown: function(e) {
        // Close results pane on escape
        if (e.key === 'Escape') {
            this.hideResultsPane();
            Tagger.Tooltip.hideTooltip();
            return;
        }
        
        // Handle keyboard navigation in search results
        if (Tagger.elements.resultsPane.style.display === 'block') {
            const resultItems = Array.from(Tagger.elements.resultsList.querySelectorAll('.result-item'));
            const itemCount = resultItems.length;
            
            if (itemCount > 0) {
                if (e.key === 'ArrowDown') {
                    e.preventDefault(); // Prevent cursor movement
                    Tagger.state.selectedResultIndex = (Tagger.state.selectedResultIndex + 1) % itemCount;
                    this.updateSelectedResult(resultItems);
                } else if (e.key === 'ArrowUp') {
                    e.preventDefault(); // Prevent cursor movement
                    Tagger.state.selectedResultIndex = (Tagger.state.selectedResultIndex - 1 + itemCount) % itemCount;
                    this.updateSelectedResult(resultItems);
                } else if (e.key === 'Enter' && Tagger.state.selectedResultIndex >= 0) {
                    e.preventDefault(); // Prevent default Enter behavior
                    resultItems[Tagger.state.selectedResultIndex].click();
                }
            }
        }
    },
    
    handleKeyup: function(e) {
        if (e.key === 'Backspace' || e.key === 'Delete') {
            this.cleanupEmptyTaggedSpans();
        }
    },
    
    handleGlobalKeydown: function(e) {
        if (e.key === 'Shift') {
            Tagger.state.shiftPressed = true;
            if (Tagger.elements.resultsPane.style.display === 'block' && 
                Tagger.state.selectedResultIndex >= 0) {
                const resultItems = Array.from(Tagger.elements.resultsList.querySelectorAll('.result-item'));
                if (resultItems.length > 0 && Tagger.state.selectedResultIndex < resultItems.length) {
                    const selectedItem = resultItems[Tagger.state.selectedResultIndex];
                    Tagger.Tooltip.showTooltip(selectedItem.dataset.body, selectedItem.dataset.file);
                }
            }
        }
    },
    
    handleGlobalKeyup: function(e) {
        if (e.key === 'Shift') {
            Tagger.state.shiftPressed = false;
            Tagger.Tooltip.hideTooltip();
        }
    },
    
    handleEditorClick: function(e) {
        // Check if we clicked on a tagged word
        if (e.target.closest('.tagged-text')) {
            const tagElement = e.target.closest('.tagged-text');
            
            // If clicking on the same tag that's already active, do nothing
            if (tagElement === Tagger.state.activeTagElement && 
                Tagger.elements.tooltip.style.display === 'block') {
                return;
            }
            
            // Set this as the active tag
            Tagger.state.activeTagElement = tagElement;
            
            // Show the tooltip for this tag
            Tagger.Tooltip.showTooltip(
                tagElement.dataset.body, 
                tagElement.dataset.file
            );
        } else {
            // If we clicked elsewhere in the editor, hide the tooltip
            Tagger.Tooltip.hideTooltip();
            Tagger.state.activeTagElement = null;
        }
    },
    
    handleGlobalClick: function(e) {
        // If clicking outside the editor and tooltip, hide tooltip and results
        if (!e.target.closest('.tooltip') && 
            !e.target.closest('#editor') && 
            !e.target.closest('.result-item')) {
            Tagger.Tooltip.hideTooltip();
            this.hideResultsPane();
            Tagger.state.activeTagElement = null;
        }
    },
    
    cleanupEmptyTaggedSpans: function() {
        const editor = Tagger.elements.editor;
        
        // Get the raw HTML content
        const rawContent = editor.innerHTML;
        const textContent = editor.textContent.trim();
        
        // More aggressive detection of "empty" state - account for various browser representations
        if (textContent === '' || 
            rawContent === '' || 
            rawContent === '<br>' || 
            rawContent === '&nbsp;' || 
            /^(\s|&nbsp;|<br\s*\/?>|<div>(<br\s*\/?>)*<\/div>)*$/i.test(rawContent)) {
                            
            // Complete reset - this is crucial to remove any hidden spans
            editor.innerHTML = '';
            
            // Also reset any saved state
            Tagger.state.currentTag = '';
            Tagger.state.lastRange = null;
            Tagger.state.activeTagElement = null;
            Tagger.state.currentTrigger = '';
            
            // Force cursor position to be reset properly
            const range = document.createRange();
            range.setStart(editor, 0);
            range.collapse(true);
            
            const selection = window.getSelection();
            selection.removeAllRanges();
            selection.addRange(range);
            
            return true;
        }
        return false;
    },
    
    findTriggerBeforeCursor: function(range) {
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
    },
    
    searchForTag: async function(tag, trigger) {
        try {
            let queryParams = `query=${encodeURIComponent(tag)}`;
            
            // Add node_type param based on trigger
            if (trigger === '/') {
                queryParams += '&node_types=Page';
            } else if (trigger === '@') {
                queryParams += '&node_types=File';
            }
            
            const response = await fetch(`${Tagger.config.base_url}/search?${queryParams}`);
            if (!response.ok) throw new Error(`HTTP error ${response.status}`);
            
            const data = await response.json();
            this.displaySearchResults(data);
        } catch (error) {
            console.error("Search error:", error);
            this.hideResultsPane();
        }
    },
    
    displaySearchResults: function(results) {
        const resultsList = Tagger.elements.resultsList;
        resultsList.innerHTML = '';
        
        if (!results || results.length === 0) {
            this.hideResultsPane();
            return;
        }
        
        results.forEach((result) => {
            const resultItem = document.createElement('div');
            resultItem.className = 'result-item';
            
            // Store data for tooltip
            resultItem.dataset.body = result.properties.body || '';
            resultItem.dataset.file = result.properties.file || '';
            
            // Create container for flex layout
            const contentContainer = document.createElement('div');
            contentContainer.style.display = 'flex';
            contentContainer.style.justifyContent = 'space-between';
            contentContainer.style.alignItems = 'center';
            contentContainer.style.width = '100%';
            
            // Name container (to allow text overflow handling)
            const nameContainer = document.createElement('div');
            nameContainer.style.overflow = 'hidden';
            nameContainer.style.textOverflow = 'ellipsis';
            nameContainer.style.whiteSpace = 'nowrap';
            nameContainer.textContent = result.properties.name || 'Unnamed';
            contentContainer.appendChild(nameContainer);
            
            // Add language tag if file exists
            if (result.properties.file) {
                const langBadge = document.createElement('span');
                langBadge.className = 'lang-badge';
                langBadge.textContent = Tagger.Tooltip.fileExtension(result.properties.file);
                langBadge.style.fontSize = '9px';
                langBadge.style.padding = '1px 4px';
                langBadge.style.backgroundColor = 'rgba(255,255,255,0.2)';
                langBadge.style.borderRadius = '3px';
                langBadge.style.marginLeft = '4px';
                langBadge.style.flexShrink = '0';
                contentContainer.appendChild(langBadge);
            }
            
            resultItem.appendChild(contentContainer);
            
            const color = Tagger.constants.nodeTypeColors[result.node_type] || '#333';
            resultItem.style.backgroundColor = color;
            
            // Add hover event for preview
            resultItem.addEventListener('mouseover', () => {
                if (!Tagger.state.shiftPressed) { // Only show tooltip on hover if shift is not pressed
                    Tagger.Tooltip.showTooltip(result.properties.body, result.properties.file);
                }
            });
            
            resultItem.addEventListener('mouseout', () => {
                if (!Tagger.state.activeTagElement && !Tagger.state.shiftPressed) {
                    Tagger.Tooltip.hideTooltip();
                }
            });
            
            // Update selection on mouse enter
            resultItem.addEventListener('mouseenter', () => {
                const resultItems = Array.from(Tagger.elements.resultsList.querySelectorAll('.result-item'));
                Tagger.state.selectedResultIndex = resultItems.indexOf(resultItem);
                this.updateSelectedResult(resultItems);
            });
            
            // Add click event
            resultItem.addEventListener('click', () => {
                this.replaceTagWithStyledSpan(result);
                this.hideResultsPane();
            });
            
            resultsList.appendChild(resultItem);
        });
        
        Tagger.elements.resultsPane.style.display = 'block';
        
        // Initialize selection to first item
        Tagger.state.selectedResultIndex = 0;
        this.updateSelectedResult(Array.from(resultsList.querySelectorAll('.result-item')));
    },
    
    updateSelectedResult: function(resultItems) {
        // Remove selection from all items
        resultItems.forEach(item => item.classList.remove('selected'));
        
        // Apply selection to current item
        if (Tagger.state.selectedResultIndex >= 0 && 
            Tagger.state.selectedResultIndex < resultItems.length) {
            resultItems[Tagger.state.selectedResultIndex].classList.add('selected');
            
            // Ensure the selected item is visible (scroll if needed)
            resultItems[Tagger.state.selectedResultIndex].scrollIntoView({ 
                behavior: 'smooth', 
                block: 'nearest' 
            });
            
            // If shift is pressed, show the tooltip for the selected item
            if (Tagger.state.shiftPressed) {
                const selectedItem = resultItems[Tagger.state.selectedResultIndex];
                Tagger.Tooltip.showTooltip(selectedItem.dataset.body, selectedItem.dataset.file);
            }
        }
    },
    
    replaceTagWithStyledSpan: function(result) {
        if (!Tagger.state.lastRange) return;
        
        // Restore the saved range
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(Tagger.state.lastRange);
        
        const range = selection.getRangeAt(0);
        const textNode = range.startContainer;
        
        if (textNode.nodeType !== Node.TEXT_NODE) return;
        
        const text = textNode.textContent;
        const cursorPos = range.startOffset;
        
        // Find the start of the hashtag
        let hashStart = -1;
        for (let i = cursorPos - 1; i >= 0; i--) {
            if (text[i] === '#' || text[i] === '/' || text[i] === '@') {
                hashStart = i;
                break;
            } else if (!/[\w\.-]/.test(text[i])) {
                break;
            }
        }
        
        if (hashStart === -1) return;
        
        // Split text into before, tag, and after
        const beforeText = text.substring(0, hashStart);
        const afterText = text.substring(cursorPos);
        
        // Create document fragment
        const fragment = document.createDocumentFragment();
        
        // Add text before the hashtag
        if (beforeText) {
            fragment.appendChild(document.createTextNode(beforeText));
        }
        
        // Create styled span for the hashtag
        const color = Tagger.constants.nodeTypeColors[result.node_type] || '#333';
        const tagSpan = document.createElement('span');
        tagSpan.className = 'tagged-text';
        tagSpan.style.color = color;
        tagSpan.style.fontWeight = 'bold';
        // FIXME: for Files, we should show the filename
        if (result.node_type === 'File' && result.properties.name) {
            tagSpan.textContent = `@${result.properties.name}`;
        } else {
            tagSpan.textContent = `${Tagger.state.currentTrigger}${Tagger.state.currentTag}`;
        }
        tagSpan.dataset.body = result.properties.body || '';
        tagSpan.dataset.file = result.properties.file || '';
        tagSpan.dataset.nodeType = result.node_type;
        
        // Add styled span to fragment
        fragment.appendChild(tagSpan);
        
        // Create a separate text node for afterText
        // If afterText is empty, use a zero-width space to ensure there's a valid cursor position
        const afterNode = document.createTextNode(afterText || '\u200B');
        fragment.appendChild(afterNode);
        
        // Replace the original text node with our new structure
        textNode.parentNode.replaceChild(fragment, textNode);
        
        // Position cursor at the beginning of the text after the tag
        const newRange = document.createRange();
        newRange.setStartAfter(afterNode);
        newRange.collapse(true);
        selection.removeAllRanges();
        selection.addRange(newRange);
        
        // Reset state
        Tagger.state.currentTag = '';
        Tagger.state.lastRange = null;
    },
    
    hideResultsPane: function() {
        Tagger.elements.resultsPane.style.display = 'none';
        Tagger.state.selectedResultIndex = -1;
    }
} })();