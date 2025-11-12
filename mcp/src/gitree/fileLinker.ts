import { Storage } from "./store/index.js";
import { LinkResult } from "./types.js";

/**
 * Links Features to File nodes in the graph based on PR file changes
 */
export class FileLinker {
  constructor(private storage: Storage) {}

  /**
   * Link files for a single feature
   */
  async linkFeature(featureId: string): Promise<LinkResult> {
    const feature = await this.storage.getFeature(featureId);
    if (!feature) {
      throw new Error(`Feature ${featureId} not found`);
    }

    console.log(`\n🔗 Linking files for feature: ${feature.name}`);

    const result = await this.storage.linkFeaturesToFiles(featureId);

    const linkedCount = result.featureFileLinks[0]?.filesLinked || 0;
    console.log(`   ✅ Linked ${linkedCount} files`);

    return result;
  }

  /**
   * Link files for all features
   */
  async linkAllFeatures(): Promise<LinkResult> {
    const features = await this.storage.getAllFeatures();

    console.log(`\n🔗 Linking files for ${features.length} features...\n`);

    const result = await this.storage.linkFeaturesToFiles();

    console.log(`\n✅ Done linking files!`);
    console.log(`   Features processed: ${result.featuresProcessed}`);
    console.log(`   Total files linked: ${result.filesLinked}`);

    // Show details for each feature
    if (result.featureFileLinks.length > 0) {
      console.log(`\n   Details:`);
      for (const link of result.featureFileLinks) {
        const feature = await this.storage.getFeature(link.featureId);
        if (feature) {
          console.log(
            `   - ${feature.name} (${link.featureId}): ${link.filesLinked} files`
          );
        }
      }
    }

    return result;
  }
}
