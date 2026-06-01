import type { AnyStepDef } from "vein";

import resolveCheckpoint from "./resolve-checkpoint.js";
import fetchChanges from "./fetch-changes.js";
import prioritizeChanges from "./prioritize-changes.js";
import isNewRepo from "./is-new-repo.js";
import cloneRepo from "./clone-repo.js";
import bootstrapExplore from "./bootstrap-explore.js";
import fetchContent from "./fetch-content.js";
import conceptDecide from "./concept-decide.js";
import applyDecision from "./apply-decision.js";
import collectResults from "./collect-results.js";
import summarizeConcept from "./summarize-concept.js";
import linkFiles from "./link-files.js";

/**
 * All in-code concepts steps, layered onto vein's core + lib steps by
 * `createRegistry`.
 */
export const conceptSteps: AnyStepDef[] = [
  resolveCheckpoint,
  fetchChanges,
  prioritizeChanges,
  isNewRepo,
  cloneRepo,
  bootstrapExplore,
  fetchContent,
  conceptDecide,
  applyDecision,
  collectResults,
  summarizeConcept,
  linkFiles,
];
