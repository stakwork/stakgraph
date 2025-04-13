// Simple emit function for custom events
export function emit(element, name, detail = {}) {
  element.dispatchEvent(new CustomEvent(name, {
    detail,
    bubbles: true,
    composed: true
  }));
}

// Constants
export const constants = {
  nodeTypeColors: {
    "Repository": "#2C5985",
    "Language": "#A35D00",
    "Directory": "#3A7336",
    "File": "#ad8cc6",
    "Import": "#8B2E2A",
    "Class": "#4A7D4A",
    "Trait": "#3B6EB5",
    "Library": "#A83333",
    "Function": "#C67000",
    "Test": "#B7940A",
    "E2etest": "#7C4A85",
    "Endpoint": "#385D8A",
    "Request": "#6B4A7A",
    "Datamodel": "#A13939",
    "Page": "#2980B9"
  }
};

// File utilities
export const fileUtils = {
  // Get file extension from path
  fileExtension(filePath) {
    if (!filePath) return null;
    const parts = filePath.split('.');
    return parts.length > 1 ? parts.pop().toLowerCase() : null;
  },
  
  // Get human-readable language name
  getLanguageFromFilePath(filePath) {
    if (!filePath) return null;
    const extension = this.fileExtension(filePath);
    if (!extension) return 'Unknown'; // Fix: Handle null extension
    
    const langMap = {
      'js': 'JavaScript',
      'jsx': 'React',
      'ts': 'TypeScript',
      'tsx': 'React TSX',
      'py': 'Python',
      'go': 'Go',
      'rs': 'Rust',
    };
    
    return langMap[extension] || extension.toUpperCase();
  },
  
  // Get highlight.js class for a file
  getHighlightJsClass(filePath) {
    if (!filePath) return '';
    
    const extension = this.fileExtension(filePath);
    const langMap = {
      'js': 'language-javascript',
      'jsx': 'language-javascript',
      'ts': 'language-typescript',
      'tsx': 'language-typescript',
      'py': 'language-python',
      'go': 'language-go',
      'rs': 'language-rust',
      // Add more mappings as needed
    };
    
    return langMap[extension] || '';
  }
};