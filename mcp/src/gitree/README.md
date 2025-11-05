# Gitree - GitHub Feature Knowledge Base

A tool that processes GitHub PRs chronologically using an LLM to organize them into conceptual features. A PR can belong to multiple features, creating a many-to-many relationship between PRs and features.

## Features

- **Intelligent Feature Extraction**: Uses Claude (Anthropic) to analyze PRs and group them into meaningful features
- **Comprehensive PR Analysis**: Fetches full PR content including diffs, comments, reviews, and commit history
- **Incremental Processing**: Remembers last processed PR to avoid reprocessing
- **Many-to-Many Relationships**: A PR can belong to multiple features
- **File-Based Storage**: Simple JSON and Markdown files for easy inspection and version control
- **CLI Interface**: Multiple commands for processing and querying the knowledge base

## Installation

```bash
# Install dependencies
yarn install

# Build the project
yarn build
```

## Prerequisites

1. **GitHub Token**: Set `GITHUB_TOKEN` environment variable with a GitHub personal access token
2. **Anthropic API Key**: Set `ANTHROPIC_API_KEY` environment variable

```bash
export GITHUB_TOKEN="your_github_token"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

## Usage

### Process a Repository

Process a GitHub repository to extract features from its PR history:

```bash
yarn gitree process <owner> <repo> [options]

# Example
yarn gitree process facebook react

# With custom storage directory
yarn gitree process facebook react --dir ./my-knowledge-base

# With explicit GitHub token
yarn gitree process facebook react --token ghp_xxxxx
```

The tool will:
1. Fetch all merged PRs chronologically
2. Skip obvious maintenance PRs (chores, dependabot, etc.)
3. Ask Claude to categorize each PR into features
4. Store results in the knowledge base directory

### List All Features

View all features in the knowledge base:

```bash
yarn gitree list-features [options]

# Example
yarn gitree list-features

# From custom directory
yarn gitree list-features --dir ./my-knowledge-base
```

### Show Feature Details

View details of a specific feature including all its PRs:

```bash
yarn gitree show-feature <featureId> [options]

# Example
yarn gitree show-feature authentication-system
```

### Show PR Details

View details of a specific PR including which features it belongs to:

```bash
yarn gitree show-pr <number> [options]

# Example
yarn gitree show-pr 1234
```

### Show Statistics

View knowledge base statistics:

```bash
yarn gitree stats [options]

# Example
yarn gitree stats
```

## Knowledge Base Structure

The knowledge base is stored on disk with this structure:

```
./knowledge-base/
  ├── metadata.json          # { lastProcessedPR: 123 }
  ├── features/
  │   ├── auth-system.json
  │   ├── google-integration.json
  │   └── payment-processing.json
  └── prs/
      ├── 1.md
      ├── 2.md
      └── 1034.md
```

### Feature JSON Format

```json
{
  "id": "auth-system",
  "name": "Authentication System",
  "description": "Complete authentication system with OAuth support, session management, and permission handling.",
  "prNumbers": [12, 15, 23, 45, 67, 89],
  "createdAt": "2023-01-15T10:30:00Z",
  "lastUpdated": "2023-03-20T14:22:00Z"
}
```

### PR Markdown Format

```markdown
# PR #1034: Add Google OAuth Support

**Merged**: 2023-02-10
**URL**: https://github.com/owner/repo/pull/1034

## Summary

Integrates Google OAuth 2.0 into the authentication system. Users can now sign in with their Google accounts using the standard OAuth flow.

---

_Part of features: `auth-system`, `google-integration`_
```

## How It Works

1. **PR Filtering**: Basic heuristics skip obvious maintenance PRs (bump, chore:, dependabot, docs:, typo, ci:)

2. **LLM Decision**: For each PR, Claude analyzes:
   - PR title and description
   - Changed files and diffs
   - Review comments and discussion
   - Commit history

   Then decides whether to:
   - Add to existing feature(s)
   - Create new feature(s)
   - Ignore (if truly trivial)

3. **Storage**: Results are saved as:
   - Features: JSON files in `features/` directory
   - PRs: Markdown files in `prs/` directory
   - Metadata: Tracking last processed PR number

## Architecture

The implementation consists of several modules:

- **types.ts**: Core TypeScript interfaces (Feature, PRRecord, LLMDecision, etc.)
- **storage.ts**: Abstract Storage class and FileSystemStore implementation
- **llm.ts**: LLMClient wrapper using Anthropic's generateObject
- **builder.ts**: StreamingFeatureBuilder - main processing logic
- **pr.ts**: Comprehensive PR content fetcher using Octokit
- **cli.ts**: Command-line interface

## Example Workflow

```bash
# 1. Process your repo
yarn gitree process myorg myrepo

# 2. See what features were discovered
yarn gitree list-features

# Output:
# 📚 Features (5 total):
#
# 🔹 Authentication System (authentication-system)
#    Complete auth with OAuth, sessions, permissions
#    PRs: 12 | Last updated: 2023-03-20
#
# 🔹 Payment Processing (payment-processing)
#    Stripe integration with webhooks and subscriptions
#    PRs: 8 | Last updated: 2023-03-15
# ...

# 3. Explore a specific feature
yarn gitree show-feature authentication-system

# 4. Check a PR's features
yarn gitree show-pr 1034

# 5. View stats
yarn gitree stats
```

## Tips

- **Start Small**: Test with a repo that has ~10-50 PRs before processing large repos
- **Incremental**: The tool remembers where it left off, so you can stop and resume
- **Version Control**: The knowledge base is just files - commit it to git!
- **Customize Prompts**: Edit `llm.ts` to adjust the system prompt and decision format

## Limitations

- **Rate Limits**: GitHub and Anthropic API rate limits apply
- **Cost**: Each PR requires 1 LLM call (costs vary by PR size)
- **Pagination**: Currently fetches max 100 PRs per request
- **Large Repos**: Very large repos may take considerable time

## Programmatic Usage

You can also use gitree as a library:

```typescript
import {
  FileSystemStore,
  LLMClient,
  StreamingFeatureBuilder
} from './src/gitree/index.js';
import { Octokit } from '@octokit/rest';

const storage = new FileSystemStore('./my-kb');
await storage.initialize();

const octokit = new Octokit({ auth: process.env.GITHUB_TOKEN });
const llm = new LLMClient('anthropic', process.env.ANTHROPIC_API_KEY);
const builder = new StreamingFeatureBuilder(storage, llm, octokit);

await builder.processRepo('facebook', 'react');

// Query the results
const features = await storage.getAllFeatures();
const authPRs = await storage.getPRsForFeature('authentication-system');
```

## Contributing

See the main project README for contribution guidelines.

## License

See the main project LICENSE file.
