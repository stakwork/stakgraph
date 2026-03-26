const KNOWN_PREFIXES = [
  "Auth error: ",
  "Validation error: ",
  "Not found: ",
  "Dependency error: ",
  "Internal error: ",
  "Other error: ",
  "Error : ",
  "Error: ",
];

function stripPrefix(msg: string): string {
  for (const prefix of KNOWN_PREFIXES) {
    if (msg.startsWith(prefix)) {
      return msg.slice(prefix.length);
    }
  }
  return msg;
}

export function cleanError(raw: string): string {
  let msg = raw.trim();

  msg = stripPrefix(msg);
  const repoLineMatch = msg.match(/Repo '[^']*':\s*(.+)/);
  if (repoLineMatch) {
    msg = stripPrefix(repoLineMatch[1].trim());
  }

  const cutIdx = msg.search(/\.\s*(Error|Failed):/);
  if (cutIdx > 0) {
    msg = msg.slice(0, cutIdx);
  }

  const newlineIdx = msg.indexOf("\n");
  if (newlineIdx > 0) {
    msg = msg.slice(0, newlineIdx);
  }

  msg = msg.trim();
  if (msg.length > 0) {
    msg = msg.charAt(0).toUpperCase() + msg.slice(1);
  }

  return msg || "An unexpected error occurred.";
}
