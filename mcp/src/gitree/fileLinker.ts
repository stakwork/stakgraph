import { Storage } from "./store/index.js";
import { LinkResult } from "./types.js";

/**
 * Links Features to File nodes in the graph based on PR file changes
 */
export class FileLinker {
  constructor(private storage: Storage) {}

  /**
   * Link files for a single feature
   * @param featureId - Feature ID (can be repo-prefixed or not)
   * @param repo - Optional repo to help locate the feature
   */
  async linkFeature(featureId: string, repo?: string): Promise<LinkResult> {
    const feature = await this.storage.getFeature(featureId, repo);
    if (!feature) {
      throw new Error(`Feature ${featureId} not found`);
    }

    console.log(`\n🔗 Linking files for feature: ${feature.name}`);

    const result = await this.storage.linkFeaturesToFiles(featureId, feature.repo);

    const link = result.featureFileLinks[0];
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
   * Link files for all features
   * @param repo - Optional repo to filter features
   */
  async linkAllFeatures(repo?: string): Promise<LinkResult> {
    const features = await this.storage.getAllFeatures(repo);

    console.log(`\n🔗 Linking files for ${features.length} features${repo ? ` in ${repo}` : ''}...\n`);

    const result = await this.storage.linkFeaturesToFiles(undefined, repo);

    console.log(`\n✅ Done linking files!`);
    console.log(`   Features processed: ${result.featuresProcessed}`);
    console.log(`   Total files linked: ${result.filesLinked}`);
    console.log(
      `   📚 ${result.filesInDocs} in documentation (importance 0.5-1.0)`
    );
    console.log(
      `   📄 ${result.filesNotInDocs} not in documentation (importance 0.0-0.49)`
    );

    // Show details for each feature
    if (result.featureFileLinks.length > 0) {
      console.log(`\n   Details:`);
      for (const link of result.featureFileLinks) {
        const feature = await this.storage.getFeature(link.featureId);
        if (feature) {
          console.log(
            `   - ${feature.name} (${link.featureId}): ${link.filesLinked} files (${link.filesInDocs} in docs, ${link.filesNotInDocs} not in docs)`
          );
        }
      }
    }

    return result;
  }
}
