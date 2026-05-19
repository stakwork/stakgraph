// EmptyState — what we render when a query succeeded but the result
// set was empty. The phrasing varies per page (no agents seen, no
// runs in window, …), so the message is a prop rather than a
// hard-coded "No data".

export function EmptyState({ children }: { children: any }) {
  return <div class="empty">{children}</div>;
}
