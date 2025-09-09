// DOM Inspector utilities for bug identification feature

import { ComponentInfo } from './types';

export interface SourceMapping {
  element: Element;
  source?: string;
  line?: string;
  column?: string;
  text?: string;
  selector?: string;
  bounds?: DOMRect;
}

export interface DebugSelection {
  elements: SourceMapping[];
  description?: string;
  coordinates?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

// Helper functions

function rectsIntersect(
  rect1: { x: number; y: number; width: number; height: number },
  rect2: { x: number; y: number; width: number; height: number }
): boolean {
  return !(
    rect1.x + rect1.width < rect2.x ||
    rect2.x + rect2.width < rect1.x ||
    rect1.y + rect1.height < rect2.y ||
    rect2.y + rect2.height < rect1.y
  );
}

export function isReactDevModeActive(): boolean {
  try {
    // Get all elements in the document
    const allElements = Array.from(document.querySelectorAll("*"));

    // Check each element for React fiber keys
    for (const element of allElements) {
      const fiberKey = Object.keys(element).find(
        (key) =>
          key.startsWith("__reactFiber$") ||
          key.startsWith("__reactInternalInstance$")
      );

      if (fiberKey) {
        return true;
      }
    }

    return false;
  } catch (error) {
    console.error("Error checking React dev mode:", error);
    return false;
  }
}

/**
 * Extract React component name from a fiber node
 */
function getComponentNameFromFiber(element: Element): ComponentInfo | null {
  try {
    // Find React fiber key
    const fiberKey = Object.keys(element).find(
      (key) =>
        key.startsWith("__reactFiber$") ||
        key.startsWith("__reactInternalInstance$")
    );

    if (!fiberKey) {
      return null;
    }

    // @ts-expect-error - Accessing React internals
    let fiber = element[fiberKey];
    let level = 0;
    const maxTraversalDepth = 10;

    while (fiber && level < maxTraversalDepth) {
      // Skip host components (DOM elements like div, span, etc.)
      if (typeof fiber.type === 'string') {
        fiber = fiber.return;
        level++;
        continue;
      }

      // Extract component name from various React patterns
      if (fiber.type) {
        let componentName: string | null = null;

        // Standard function/class components
        if (fiber.type.displayName) {
          componentName = fiber.type.displayName;
        } else if (fiber.type.name) {
          componentName = fiber.type.name;
        }
        // React.forwardRef components
        else if (fiber.type.render) {
          componentName = fiber.type.render.displayName || fiber.type.render.name || 'ForwardRef';
        }
        // React.memo components
        else if (fiber.type.type) {
          componentName = fiber.type.type.displayName || fiber.type.type.name || 'Memo';
        }
        // React.lazy components
        else if (fiber.type._payload && fiber.type._payload._result) {
          componentName = fiber.type._payload._result.name || 'LazyComponent';
        }
        // Context consumers/providers
        else if (fiber.type._context) {
          componentName = fiber.type._context.displayName || 'Context';
        }
        // Handle Fragment
        else if (fiber.type === Symbol.for('react.fragment')) {
          fiber = fiber.return;
          level++;
          continue;
        }

        if (componentName) {
          return {
            name: componentName,
            level: level,
            type: typeof fiber.type === 'function' ? 'function' : 'class'
          };
        }
      }

      fiber = fiber.return;
      level++;
    }

    return null;
  } catch (error) {
    console.error("Error extracting component name:", error);
    return null;
  }
}

/**
 * Extract React fiber debug source information from a DOM element
 */
function extractReactDebugSource(
  element: Element
): { fileName?: string; lineNumber?: number; columnNumber?: number } | null {
  try {
    // Find React fiber key
    const fiberKey = Object.keys(element).find(
      (key) =>
        key.startsWith("__reactFiber$") ||
        key.startsWith("__reactInternalInstance$")
    );

    if (!fiberKey) {
      return null;
    }

    // @ts-expect-error - Accessing React internals
    let fiber = element[fiberKey];
    let level = 0;

    // Get max traversal depth from config, default to 10
    const maxTraversalDepth = Number((window as any).STAKTRAK_CONFIG?.maxTraversalDepth) || 10;

    // Helper to extract source from an object
    const extractSource = (
      source: {
        fileName?: string;
        lineNumber?: number;
        columnNumber?: number;
      } | null
    ) => {
      if (!source) return null;
      return {
        fileName: source.fileName,
        lineNumber: source.lineNumber,
        columnNumber: source.columnNumber,
      };
    };

    // Traverse up the fiber tree to find debug source
    while (fiber && level < maxTraversalDepth) {
      // Check various locations where source info might be stored
      const source =
        fiber._debugSource ||
        fiber.memoizedProps?.__source ||
        fiber.pendingProps?.__source;

      if (source) {
        return extractSource(source);
      }

      // Go up the component tree
      fiber = fiber.return;
      level++;
    }

    return null;
  } catch (error) {
    console.error("Error extracting React debug source:", error);
    return null;
  }
}

type DebugMsgData = {
  messageId: string;
  coordinates: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
};

/**
 * Initialize debug message listener for when this app is loaded in an iframe
 */
export function debugMsg(data: DebugMsgData) {
  const { messageId, coordinates } = data;

  try {
    // Handle element finding directly since we're inside the target document

    // For direct DOM access (since we're inside the iframe), we need to search differently
    const sourceFiles: Array<{
      file: string;
      lines: number[];
      context?: string;
      message?: string;
      componentNames?: Array<{
        name: string;
        level: number;
        type: string;
        element: string;
      }>;
    }> = [];
    const processedFiles = new Map<string, Set<number>>();
    
    // Track component names found
    const componentNames: Array<{
      name: string;
      level: number;
      type: string;
      element: string;
    }> = [];
    const processedComponents = new Set<string>();

    // Get element at point or elements in region
    let elementsToProcess: Element[] = [];

    if (coordinates.width === 0 && coordinates.height === 0) {
      // Click mode - get element at point
      const element = document.elementFromPoint(coordinates.x, coordinates.y);

      if (element) {
        elementsToProcess = [element];
        // Also include parents up to body
        let parent = element.parentElement;
        while (parent && parent !== document.body) {
          elementsToProcess.push(parent);
          parent = parent.parentElement;
        }
      }
    } else {
      // Selection mode - get all elements in the rectangle
      const allElements = document.querySelectorAll("*");
      elementsToProcess = Array.from(allElements).filter((el) => {
        const rect = el.getBoundingClientRect();
        return rectsIntersect(
          {
            x: rect.left,
            y: rect.top,
            width: rect.width,
            height: rect.height,
          },
          coordinates
        );
      });
    }

    // Process each element to extract debug source
    for (const element of elementsToProcess) {
      // First try to extract React component name
      const componentInfo = getComponentNameFromFiber(element);
      if (componentInfo && !processedComponents.has(componentInfo.name)) {
        processedComponents.add(componentInfo.name);
        componentNames.push({
          name: componentInfo.name,
          level: componentInfo.level,
          type: componentInfo.type,
          element: element.tagName.toLowerCase()
        });
      }
      
      // First check for data attributes (if using react-dev-inspector or similar)
      const dataSource =
        element.getAttribute("data-source") ||
        element.getAttribute("data-inspector-relative-path");
      const dataLine =
        element.getAttribute("data-line") ||
        element.getAttribute("data-inspector-line");

      if (dataSource && dataLine) {
        const lineNum = parseInt(dataLine, 10);
        if (
          !processedFiles.has(dataSource) ||
          !processedFiles.get(dataSource)?.has(lineNum)
        ) {
          if (!processedFiles.has(dataSource)) {
            processedFiles.set(dataSource, new Set());
          }
          processedFiles.get(dataSource)!.add(lineNum);

          // Find existing file entry or create new one
          let fileEntry = sourceFiles.find((f) => f.file === dataSource);
          if (!fileEntry) {
            fileEntry = { file: dataSource, lines: [] };
            sourceFiles.push(fileEntry);
          }
          fileEntry.lines.push(lineNum);
        }
      } else {
        // Fall back to React fiber debug source
        const debugSource = extractReactDebugSource(element);
        if (debugSource && debugSource.fileName && debugSource.lineNumber) {
          const fileName = debugSource.fileName;
          const lineNum = debugSource.lineNumber;

          if (
            !processedFiles.has(fileName) ||
            !processedFiles.get(fileName)?.has(lineNum)
          ) {
            if (!processedFiles.has(fileName)) {
              processedFiles.set(fileName, new Set());
            }
            processedFiles.get(fileName)!.add(lineNum);

            // Find existing file entry or create new one
            let fileEntry = sourceFiles.find((f) => f.file === fileName);
            if (!fileEntry) {
              fileEntry = { file: fileName, lines: [] };
              sourceFiles.push(fileEntry);
            }
            fileEntry.lines.push(lineNum);

            // Add context about the element
            const tagName = element.tagName.toLowerCase();
            const className = element.className
              ? `.${element.className.split(" ")[0]}`
              : "";
            fileEntry.context = `${tagName}${className}`;
          }
        }
      }
    }

    // Sort lines within each file
    sourceFiles.forEach((file) => {
      file.lines.sort((a, b) => a - b);
    });
    
    // Helper function to format components for chat display
    const formatComponentsForChat = (components: typeof componentNames): string | undefined => {
      if (components.length === 0) return undefined;

      // Sort by level (closest to clicked element first) and take top 3
      const sortedComponents = components
        .sort((a, b) => a.level - b.level)
        .slice(0, 3);

      const componentLines = sortedComponents.map(c => {
        const nameToUse = c.name || 'Unknown';
        return `&lt;${nameToUse}&gt; (${c.level} level${c.level !== 1 ? 's' : ''} up)`;
      });

      return "React Components Found:\n" + componentLines.join("\n");
    };
    
    // Enhanced fallback with component information
    if (sourceFiles.length === 0) {
      if (componentNames.length > 0) {
        // We found components but no source mapping
        const formattedMessage = formatComponentsForChat(componentNames);
        sourceFiles.push({
          file: "React component detected",
          lines: [],
          context: `Components found: ${componentNames.map(c => c.name).join(", ")}`,
          componentNames: componentNames,
          message: formattedMessage
        });
      } else {
        // No components or source mapping found
        sourceFiles.push({
          file: "No React components detected",
          lines: [],
          context: "The selected element may not be a React component or may be a native DOM element",
          message: "Try selecting an interactive element like a button or link"
        });
      }
    } else {
      // Add component names to existing source files
      sourceFiles.forEach(file => {
        if (!file.componentNames && componentNames.length > 0) {
          file.componentNames = componentNames;
          // Add formatted message to existing files too
          const formattedMessage = formatComponentsForChat(componentNames);
          if (formattedMessage) {
            file.message = formattedMessage;
          }
        }
      });
    }

    // Send response back to parent frame
    window.parent.postMessage(
      {
        type: "staktrak-debug-response",
        messageId,
        success: true,
        sourceFiles,
      },
      "*"
    );
  } catch (error) {
    console.error("Error processing debug request:", error);

    // Send error response
    window.parent.postMessage(
      {
        type: "staktrak-debug-response",
        messageId,
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
        sourceFiles: [],
      },
      "*"
    );
  }
}
