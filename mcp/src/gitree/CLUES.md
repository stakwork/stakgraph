# Clue System: Automatic Discovery & Linking

The Clue system discovers architectural patterns, utilities, and abstractions from your codebase and automatically links them to relevant features.

## Architecture

### Clue Structure
```typescript
interface Clue {
  id: string;                    // Slug from title
  featureId: string;             // Where it was discovered (provenance)
  type: ClueType;                // utility, pattern, abstraction, etc.
  title: string;
  content: string;               // WHY, WHEN, CONTEXT
  entities: ClueEntities;        // Functions, classes, types, endpoints
  files: string[];               // Associated files
  keywords: string[];
  embedding?: number[];          // Vector for semantic search
  relatedFeatures: string[];     // Features this clue is relevant to
  // ... other fields
}
```

### Graph Structure
```
(Clue)-[:RELEVANT_TO]->(Feature)
(Clue)-[:RELEVANT_TO]->(Feature)
(Clue)-[:RELEVANT_TO]->(Feature)
```

- **No `BELONGS_TO` edge** - only `RELEVANT_TO` edges
- `featureId` property stores where the clue was discovered
- `relatedFeatures` property + edges track all relevant features

## How It Works

When you run `analyze-clues`, the system automatically performs both discovery and linking:

### Phase 1: Discover Clues
1. Get files from feature (PRs + commits)
2. Use `get_context` agent to analyze codebase
3. Extract 5-10 clues per run (max 40 per feature)
4. Generate embedding for each clue title
5. Initially link to discovering feature only (`relatedFeatures: [featureId]`)

### Phase 2: Auto-Link (Automatic)
1. Load all features (names, descriptions, docs)
2. Load all clues (titles, content, keywords, entities)
3. Use `get_context` agent to determine relevance
4. Create `RELEVANT_TO` edges based on:
   - Semantic similarity (embeddings)
   - Keyword matches
   - Entity usage across features
   - File overlap

### Run it:

**API:**
```bash
# Analyze and auto-link (default)
curl -X POST "http://localhost:3355/gitree/analyze-clues?owner=stakwork&repo=hive"

# Single feature
curl -X POST "http://localhost:3355/gitree/analyze-clues?owner=stakwork&repo=hive&feature_id=quick-ask"

# Skip auto-linking
curl -X POST "http://localhost:3355/gitree/analyze-clues?owner=stakwork&repo=hive&auto_link=false"
```

**CLI:**
```bash
# Analyze all features (auto-links by default)
yarn gitree analyze-clues -r /path/to/repo

# Single feature
yarn gitree analyze-clues quick-ask -r /path/to/repo

# Skip auto-linking
yarn gitree analyze-clues -r /path/to/repo --no-link
```

### Example Output:
```
📚 Analyzing clues for 8 features...

[1/8] Processing: Quick Ask (quick-ask)
   ✨ Created clue: Streaming AI Response Pattern with AI SDK [pattern]
   ✨ Created clue: Middleware-Based Authentication Pattern [pattern]
   ✨ Created clue: Mock Factory Pattern for Testing AI Streams [pattern]
   📊 Created 9 new clues (total: 9)
   ✅ Analysis complete

✅ Done analyzing all features!
   Total token usage: 45,234

🔗 Automatically linking clues to relevant features...

[1-20/27] Processing batch...
   🤖 Analyzing 20 clues for relevance...
   🔗 Linked "Streaming AI Response Pattern" to 3 feature(s)
   🔗 Linked "Middleware-Based Authentication Pattern" to 5 feature(s)
   🔗 Linked "Mock Factory Pattern" to 2 feature(s)

✅ Total usage (analysis + linking): 52,678
```

## Manual Linking (Optional)

If you need to re-link clues without re-analyzing:

**API:**
```bash
curl -X POST "http://localhost:3355/gitree/link-clues?owner=stakwork&repo=hive"
```

**CLI:**
```bash
yarn gitree link-clues -r /path/to/repo
```

## Querying Clues

### By Relevance Search (Semantic)
```bash
# Search with embeddings + keywords + centrality
curl -X POST "http://localhost:3355/gitree/search-clues" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication middleware patterns",
    "limit": 5
  }'

# CLI
yarn gitree search-clues "authentication middleware" -l 10
```

### By Feature (Graph Query)
```cypher
// Get all clues for a feature
MATCH (c:Clue)-[:RELEVANT_TO]->(f:Feature {id: "quick-ask"})
RETURN c, f

// Find features using a specific clue
MATCH (c:Clue {id: "middleware-auth-pattern"})-[:RELEVANT_TO]->(f:Feature)
RETURN f.name, f.description

// Cross-cutting concerns (clues relevant to many features)
MATCH (c:Clue)-[:RELEVANT_TO]->(f:Feature)
WITH c, count(f) as featureCount
WHERE featureCount >= 3
RETURN c.title, c.type, featureCount
ORDER BY featureCount DESC
```

## Workflow

### Initial Setup
```bash
# 1. Process PRs/commits into features
curl -X POST "http://localhost:3355/gitree/process?owner=stakwork&repo=hive"

# 2. Summarize features
curl -X POST "http://localhost:3355/gitree/summarize-all?owner=stakwork&repo=hive"

# 3. Discover clues (automatically links!)
curl -X POST "http://localhost:3355/gitree/analyze-clues?owner=stakwork&repo=hive"
```

### Incremental Updates
```bash
# After new PRs are added:
# 1. Process new PRs
curl -X POST "http://localhost:3355/gitree/process?owner=stakwork&repo=hive"

# 2. Analyze clues (automatically re-links all)
curl -X POST "http://localhost:3355/gitree/analyze-clues?owner=stakwork&repo=hive"
```

## Linking Principles

The agent uses these rules when linking:

1. **Cross-cutting concerns** (auth, logging, error handling) → Many features (5-10)
2. **Domain-specific utilities** (payment processing, email) → 1-3 related features
3. **Generic patterns** (state management, data flow) → 3-5 features where used
4. **Feature-specific implementations** → 1-2 features only

**Quality over quantity** - Only link clues that genuinely help understand the feature.

## Example: "Middleware-Based Authentication Pattern"

### After Discovery (Phase 1)
```
featureId: "quick-ask"               // Discovered here
relatedFeatures: ["quick-ask"]       // Initially linked to discovery feature
```

Graph:
```
(Clue: middleware-auth-pattern)-[:RELEVANT_TO]->(Feature: quick-ask)
```

### After Auto-Linking (Phase 2)
```
featureId: "quick-ask"               // Still shows where discovered
relatedFeatures: [                   // Now linked to all relevant features
  "quick-ask",         // Where discovered
  "user-management",   // Uses auth middleware
  "api-gateway",       // Implements auth checks
  "workspace-context", // Requires authentication
  "admin-dashboard"    // Protected routes
]
```

Graph:
```
(Clue: middleware-auth-pattern)-[:RELEVANT_TO]->(Feature: quick-ask)
(Clue: middleware-auth-pattern)-[:RELEVANT_TO]->(Feature: user-management)
(Clue: middleware-auth-pattern)-[:RELEVANT_TO]->(Feature: api-gateway)
(Clue: middleware-auth-pattern)-[:RELEVANT_TO]->(Feature: workspace-context)
(Clue: middleware-auth-pattern)-[:RELEVANT_TO]->(Feature: admin-dashboard)
```

## Benefits

1. **Discovery Context**: Know where patterns originated
2. **Cross-Feature Reuse**: Find patterns used across features
3. **Semantic Search**: Find relevant patterns by meaning, not just keywords
4. **Architectural Insights**: Understand which features share patterns
5. **Incremental Updates**: Re-link without re-discovering

## File References

- **Types**: `mcp/src/gitree/types.ts`
- **Discovery**: `mcp/src/gitree/clueAnalyzer.ts`
- **Linking**: `mcp/src/gitree/clueLinker.ts`
- **Storage**: `mcp/src/gitree/store/graphStorage.ts`
- **Routes**: `mcp/src/gitree/routes.ts`
- **CLI**: `mcp/src/gitree/cli.ts`
