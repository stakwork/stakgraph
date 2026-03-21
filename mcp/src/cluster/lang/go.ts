import { LangConfig } from './types.js';

const config: LangConfig = {
  genericFolders: [
    'cmd', 'pkg', 'internal', 'vendor', 'api', 'server', 'client',
    'handlers', 'middleware', 'models', 'routes', 'services', 'utils',
  ],
  genericFilenames: [
    'main', 'handler', 'server', 'client', 'utils', 'helpers', 'config',
  ],
  isDynamicSegment: () => false,
};

export default config;
