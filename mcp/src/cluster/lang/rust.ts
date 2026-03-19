import { LangConfig } from './types.js';

const config: LangConfig = {
  genericFolders: [
    'src', 'lib', 'bin', 'tests', 'benches', 'examples',
  ],
  genericFilenames: [
    'mod', 'lib', 'main', 'utils', 'error', 'config',
  ],
  isDynamicSegment: () => false,
};

export default config;
