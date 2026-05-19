import { Link } from "wouter-preact";

export function NotFound() {
  return (
    <div class="empty">
      <h1 style="margin-bottom: 12px">Not found</h1>
      <p>
        That page doesn't exist. <Link href="/">Back to the dashboard</Link>.
      </p>
    </div>
  );
}
