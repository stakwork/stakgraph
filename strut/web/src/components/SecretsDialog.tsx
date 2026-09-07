// ── Secrets Dialog ──────────────────────────────────────────────────────────
// Manage deployment-scoped credentials (API keys, tokens, service-account
// JSON). Values are write-only: the list shows NAMES + metadata only — the
// server never returns a stored value. Steps read these via
// `ctx.services.secrets.get("NAME")`.

import { useEffect, useState } from "preact/hooks";
import * as api from "../api";

const NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;

export function SecretsDialog(props: { onClose: () => void }) {
  const [secrets, setSecrets] = useState<api.SecretInfo[]>([]);
  const [name, setName] = useState("");
  const [value, setValue] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = async () => {
    try {
      setSecrets(await api.listSecrets());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const handleSave = async () => {
    if (!NAME_RE.test(name)) {
      setError("Name: letters, digits, underscore (not starting with a digit)");
      return;
    }
    if (!value) {
      setError("Value is required");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await api.setSecret(name, value);
      setName("");
      setValue("");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async (n: string) => {
    setBusy(true);
    setError("");
    try {
      await api.deleteSecret(n);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const existingNames = new Set(secrets.map((s) => s.name));

  return (
    <div
      class="dialog-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) props.onClose();
      }}
    >
      <div class="dialog">
        <div class="dialog-title">Secrets</div>
        <div class="dialog-hint" style="margin-bottom:12px;">
          Deployment-wide credentials read by steps via{" "}
          <code>ctx.services.secrets.get("NAME")</code>. Values are write-only —
          they're never shown again after saving.
        </div>

        <div class="secret-list">
          {secrets.length === 0 && (
            <div class="secret-empty">No secrets yet.</div>
          )}
          {secrets.map((s) => (
            <div class="secret-row" key={s.name}>
              <span class="secret-name">{s.name}</span>
              <span class="secret-meta">
                updated {new Date(s.updatedAt).toLocaleDateString()}
              </span>
              <button
                class="btn secret-del"
                disabled={busy}
                onClick={() => handleDelete(s.name)}
                title="Delete"
              >
                ✕
              </button>
            </div>
          ))}
        </div>

        <div class="dialog-field">
          <label>Name</label>
          <input
            type="text"
            value={name}
            placeholder="GITHUB_TOKEN"
            onInput={(e) => {
              setName((e.target as HTMLInputElement).value);
              setError("");
            }}
          />
        </div>
        <div class="dialog-field">
          <label>Value</label>
          <textarea
            value={value}
            rows={3}
            placeholder="paste secret value (or service-account JSON)"
            onInput={(e) => {
              setValue((e.target as HTMLTextAreaElement).value);
              setError("");
            }}
          />
          {name && existingNames.has(name) && (
            <div class="dialog-hint">
              "{name}" exists — saving overwrites it.
            </div>
          )}
        </div>

        {error && (
          <div style="color:var(--danger);font-size:12px;margin-bottom:8px;">
            {error}
          </div>
        )}
        <div class="dialog-actions">
          <button class="btn" onClick={props.onClose}>
            Close
          </button>
          <button class="btn btn-primary" disabled={busy} onClick={handleSave}>
            Save secret
          </button>
        </div>
      </div>
    </div>
  );
}
