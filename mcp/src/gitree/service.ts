import { GraphStorage } from "./store/index.js";

export interface ConceptSummary {
  id: string;
  repo?: string;
  ref_id?: string;
  name: string;
  description: string;
  prCount: number;
  commitCount: number;
  lastUpdated: string;
  hasDocumentation: boolean;
}

export interface ConceptDocumentation {
  id: string;
  name: string;
  description: string;
  documentation?: string;
}

export async function listConcepts(repo?: string): Promise<{
  concepts: ConceptSummary[];
  total: number;
}> {
  const storage = new GraphStorage();
  await storage.initialize();

  const concepts = await storage.getAllConcepts(repo);

  return {
    concepts: concepts.map((f) => ({
      id: f.id,
      repo: f.repo,
      ref_id: f.ref_id,
      name: f.name,
      description: f.description,
      prCount: f.prNumbers.length,
      commitCount: (f.commitShas || []).length,
      lastUpdated: f.lastUpdated.toISOString(),
      hasDocumentation: !!f.documentation,
    })),
    total: concepts.length,
  };
}

export async function getConceptDocumentation(
  conceptId: string,
  repo?: string
): Promise<ConceptDocumentation | null> {
  const storage = new GraphStorage();
  await storage.initialize();

  const concept = await storage.getConcept(conceptId, repo);

  if (!concept) {
    return null;
  }

  return {
    id: concept.id,
    name: concept.name,
    description: concept.description,
    documentation: concept.documentation,
  };
}
