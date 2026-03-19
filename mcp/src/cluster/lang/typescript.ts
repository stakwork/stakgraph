import { LangConfig } from './types.js';

const config: LangConfig = {
  genericFolders: [
    'src', 'lib', 'core', 'utils', 'common', 'shared', 'helpers',
    'queries', 'mutations', 'hooks', 'components', 'pages', 'views',
    'services', 'store', 'stores', 'api', 'types', 'interfaces',
  ],
  genericFilenames: [
    'route', 'page', 'index', 'layout', 'loading', 'error', 'not-found',
    'middleware', 'handler', 'mod', 'main', 'app',
  ],
  isDynamicSegment: (s: string) => /^\[.+\]$/.test(s),
};

export default config;
