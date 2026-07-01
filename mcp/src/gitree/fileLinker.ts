import { Storage } from "./store/index.js";
import { LinkResult } from "./types.js";

/**
 * Links Concepts to File nodes in the graph based on PR file changes
 */
export class FileLinker {
  constructor(private storage: Storage) {}

  /**
   * Link files for a single concept
   * @param conceptId - Concept ID (can be repo-prefixed or not)
   * @param repo - Optional repo to help locate the concept
   */
  async linkConcept(conceptId: string, repo?: string): Promise<LinkResult> {
    const concept = await this.storage.getConcept(conceptId, repo);
    if (!concept) {
      throw new Error(`Concept ${conceptId} not found`);
    }

    console.log(`\n🔗 Linking files for concept: ${concept.name}`);

    const result = await this.storage.linkConceptsToFiles(conceptId, concept.repo);

    const link = result.conceptFileLinks[0];
    if (link) {
      console.log(`   ✅ Linked ${link.filesLinked} files`);
      console.log(
        `      📚 ${link.filesInDocs} in documentation (importance 0.5-1.0)`
      );
      console.log(
        `      📄 ${link.filesNotInDocs} not in documentation (importance 0.0-0.49)`
      );
    }

    return result;
  }

  /**
   * Link files for all concepts
   * @param repo - Optional repo to filter concepts
   */
  async linkAllConcepts(repo?: string): Promise<LinkResult> {
    const concepts = await this.storage.getAllConcepts(repo);

    console.log(`\n🔗 Linking files for ${concepts.length} concepts${repo ? ` in ${repo}` : ''}...\n`);

    const result = await this.storage.linkConceptsToFiles(undefined, repo);

    // Also create direct PullRequest -> File edges (deterministic, repo-scoped).
    // This runs in the same bulk pass so both normal ingestion and manual
    // re-linking (backfill) keep PR->File edges in sync with no extra wiring.
    const prLink = await this.storage.linkPRsToFiles(repo);
    result.prsProcessed = prLink.prsProcessed;
    result.prFileEdges = prLink.edgesLinked;

    console.log(`\n✅ Done linking files!`);
    console.log(`   Concepts processed: ${result.conceptsProcessed}`);
    console.log(`   Total files linked: ${result.filesLinked}`);
    console.log(
      `   Direct PR→File edges: ${prLink.edgesLinked} across ${prLink.prsProcessed} PRs`
    );
    console.log(
      `   📚 ${result.filesInDocs} in documentation (importance 0.5-1.0)`
    );
    console.log(
      `   📄 ${result.filesNotInDocs} not in documentation (importance 0.0-0.49)`
    );

    // Show details for each concept
    if (result.conceptFileLinks.length > 0) {
      console.log(`\n   Details:`);
      for (const link of result.conceptFileLinks) {
        const concept = await this.storage.getConcept(link.conceptId);
        if (concept) {
          console.log(
            `   - ${concept.name} (${link.conceptId}): ${link.filesLinked} files (${link.filesInDocs} in docs, ${link.filesNotInDocs} not in docs)`
          );
        }
      }
    }

    return result;
  }
}
