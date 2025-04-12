// Tooltip module - handles tooltips and code highlighting
Tagger.Tooltip = (function() { return {
    init: function() {
        // Nothing to initialize for now
    },
    
    showTooltip: function(bodyText, filePath) {
        const tooltip = Tagger.elements.tooltip;
        
        if (!bodyText) {
            tooltip.innerHTML = '<div class="file-info"><span class="file-path">No content available</span><div class="actions"><div class="untag-btn">Untag</div><div class="close-btn">×</div></div></div>';
            tooltip.style.display = 'block';
            this.setupTooltipButtons();
            return;
        }
        
        // Create file info header
        const fileInfo = document.createElement('div');
        fileInfo.className = 'file-info';
        
        const filePart = document.createElement('span');
        filePart.className = 'file-path';
        filePart.textContent = filePath || 'Unknown file';
        fileInfo.appendChild(filePart);
        
        // Actions container
        const actions = document.createElement('div');
        actions.className = 'actions';
        
        // Add untag button
        const untagBtn = document.createElement('div');
        untagBtn.className = 'untag-btn';
        untagBtn.textContent = 'Untag';
        actions.appendChild(untagBtn);
        
        // Add close button
        const closeBtn = document.createElement('div');
        closeBtn.className = 'close-btn';
        closeBtn.textContent = '×';
        actions.appendChild(closeBtn);
        
        fileInfo.appendChild(actions);
        
        // Add language badge
        if (filePath) {
            const lang = this.getLanguageFromFilePath(filePath);
            if (lang) {
                const langBadge = document.createElement('span');
                langBadge.className = 'file-type';
                langBadge.textContent = lang;
                filePart.appendChild(langBadge);
            }
        }
        
        // Create pre and code elements for syntax highlighting
        const pre = document.createElement('pre');
        const code = document.createElement('code');
        
        // If we have a file path, try to set the language class
        if (filePath) {
            const langClass = this.getHighlightJsClass(filePath);
            if (langClass) {
                code.className = langClass;
            }
        }
        
        // Show the entire code body (not just first 15 lines)
        code.textContent = bodyText;
        pre.appendChild(code);
        
        // Clear tooltip and add new content
        tooltip.innerHTML = '';
        tooltip.appendChild(fileInfo);
        tooltip.appendChild(pre);
        
        // Apply syntax highlighting
        hljs.highlightElement(code);
        
        // Show tooltip
        tooltip.style.display = 'block';
        
        // Setup tooltip buttons
        this.setupTooltipButtons();
    },
    
    setupTooltipButtons: function() {
        const tooltip = Tagger.elements.tooltip;
        
        // Setup close button
        const closeBtn = tooltip.querySelector('.close-btn');
        if (closeBtn) {
            closeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.hideTooltip();
            });
        }
        
        // Setup untag button
        const untagBtn = tooltip.querySelector('.untag-btn');
        if (untagBtn) {
            untagBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.untagElement();
                this.hideTooltip();
            });
        }
    },
    
    untagElement: function() {
        if (!Tagger.state.activeTagElement) return;

        // Get the text content without the # symbol
        const tagText = Tagger.state.activeTagElement.textContent;

        // Create a text node to replace the tag span
        const textNode = document.createTextNode(tagText);

        // Replace the tag span with the text node
        Tagger.state.activeTagElement.parentNode.replaceChild(textNode, Tagger.state.activeTagElement);

        // Reset active tag
        Tagger.state.activeTagElement = null;
    },
    
    hideTooltip: function() {
        Tagger.elements.tooltip.style.display = 'none';
    },
    
    fileExtension: function(filePath) {
        if (!filePath) return null;
        return filePath.split('.').pop().toLowerCase();
    },
    
    getLanguageFromFilePath: function(filePath) {
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
    },
    
    getHighlightJsClass: function(filePath) {
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
    }
} })();