/**
 * repo_lock.ts — per-repo queueing mutex
 *
 * `withRepoLock(repoDir, fn)` serializes all callers for the same `repoDir`
 * strictly sequentially: each invocation waits for the *full settlement* of
 * the previous one before running `fn`.  This is deliberately different from
 * the de-duplication cache pattern in clone.ts's old `cloneLocks` map, which
 * handed a second caller the *same promise and result* as the first — that
 * shape would make two concurrent PR runs silently share a worktree/branch.
 *
 * The lock is also shared by clone.ts (replacing its module-private
 * `cloneLocks` map) so that `doCloneOrUpdate`'s `fs.rmSync` calls are
 * serialized against `acquireWorktree` / `releaseWorktree`.
 */

const locks = new Map<string, Promise<unknown>>();

/**
 * Run `fn` exclusively for `repoDir`.  Every caller for the same `repoDir`
 * runs strictly after the previous caller's promise has fully settled
 * (resolved OR rejected).  Callers for different `repoDir` keys run
 * concurrently with no coordination.
 */
export function withRepoLock<T>(
  repoDir: string,
  fn: () => Promise<T>
): Promise<T> {
  // Chain behind whatever is currently running for this key.
  // We always catch the predecessor so a rejection does not propagate to us —
  // each caller is independently responsible for its own result.
  const previous = locks.get(repoDir) ?? Promise.resolve();

  const next: Promise<T> = previous.then(
    () => fn(),
    () => fn() // predecessor failed — we still run
  );

  // Store the chained promise (without propagating our rejection to successors
  // — they should still run even if we fail).
  locks.set(repoDir, next.then(
    () => {},
    () => {}
  ));

  // Clean up once the chain is idle so the map does not grow unboundedly.
  next.then(
    () => { if (locks.get(repoDir) === next) locks.delete(repoDir); },
    () => { if (locks.get(repoDir) === next) locks.delete(repoDir); }
  );

  return next;
}
