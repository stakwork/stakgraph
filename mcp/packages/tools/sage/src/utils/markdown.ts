/**
 * Extracts codespace URL from markdown links in text
 * Looks for patterns like [codespace url](https://example.com)
 * The text in brackets is case-insensitive and ignores spaces
 */
export function extractCodespaceUrl(text: string): string | null {
  // Regex to match markdown links: [text](url)
  const markdownLinkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;

  let match;
  while ((match = markdownLinkRegex.exec(text)) !== null) {
    const linkText = match[1];
    const url = match[2];

    // Normalize the link text: remove spaces and convert to lowercase
    const normalizedText = linkText.replace(/\s+/g, "").toLowerCase();

    // Check if it matches "codespaceurl"
    if (normalizedText === "codespaceurl") {
      return url.trim();
    }
  }

  return null;
}
