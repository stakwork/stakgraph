export interface LangConfig {
  genericFolders: string[];
  genericFilenames: string[];
  isDynamicSegment: (segment: string) => boolean;
}
