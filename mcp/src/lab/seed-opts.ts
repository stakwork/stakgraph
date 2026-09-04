import type { PublishByContentOptions } from "vein";

/**
 * How every lab seeder reconciles its committed templates into the workspace
 * at boot. Content-hash keyed: a template whose hash the workspace has never
 * seen publishes the next version and activates it (a real code change wins);
 * a template whose hash is already a known version is a NO-OP even when that
 * version isn't active — so an edit made through the vein UI/API (or by the
 * authoring agent) stays active across restarts until the committed template
 * itself changes. Porting the winner back into the committed template is
 * still the only way to propagate it to other instances.
 */
export const SEED_OPTS: PublishByContentOptions = { reactivateKnown: false };
