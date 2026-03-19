import { LangConfig } from './types.js';

const config: LangConfig = {
  genericFolders: [
    'src', 'lib', 'tests', 'utils', 'common', 'core', 'helpers',
    'models', 'views', 'serializers', 'services', 'api', 'schemas',
  ],
  genericFilenames: [
    '__init__', 'main', 'utils', 'helpers', 'views', 'models',
    'serializers', 'urls', 'admin', 'apps', 'config', 'settings',
  ],
  isDynamicSegment: () => false,
};

export default config;
