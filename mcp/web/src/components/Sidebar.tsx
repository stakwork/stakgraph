import { useState, useEffect, useMemo } from "react";
import { ChevronDown, BookOpen, Lightbulb } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useApi } from "@/hooks/useApi";
import type { Doc, FeaturesResponse, FeatureSummary } from "@/types";

// GET /docs returns: [ { "repo-name": { documentation: "..." } }, ... ]
type DocsResponse = Array<Record<string, { documentation: string }>>;

function parseDocs(raw: DocsResponse | null): Doc[] {
  if (!raw) return [];
  return raw.map((entry) => {
    const repoName = Object.keys(entry)[0];
    return { repoName, documentation: entry[repoName].documentation };
  });
}

function getRepoFromFeatureId(id: string): string {
  const parts = id.split("/");
  return parts.slice(0, 2).join("/");
}

interface SidebarProps {
  activeItemKey: string | null;
  onDocClick: (repoName: string, documentation: string) => void;
  onConceptClick: (id: string, name: string, description: string) => void;
}

export function Sidebar({
  activeItemKey,
  onDocClick,
  onConceptClick,
}: SidebarProps) {
  const { data: rawDocs, loading: isDocsLoading } = useApi<DocsResponse>("/docs");
  const { data: featuresData, loading: isConceptsLoading } =
    useApi<FeaturesResponse>("/gitree/features");

  const docs = useMemo(() => parseDocs(rawDocs), [rawDocs]);
  const features = featuresData?.features ?? [];

  const [isDocsExpanded, setIsDocsExpanded] = useState(true);
  const [isConceptsExpanded, setIsConceptsExpanded] = useState(true);
  const [expandedRepoGroups, setExpandedRepoGroups] = useState<
    Record<string, boolean>
  >({});

  // Seed new repo groups (default to expanded), preserve existing toggle state
  useEffect(() => {
    const newKeys = features.reduce<Record<string, boolean>>((acc, f) => {
      const repo = f.repo || getRepoFromFeatureId(f.id);
      if (!(repo in expandedRepoGroups) && !(repo in acc)) {
        acc[repo] = true;
      }
      return acc;
    }, {});
    if (Object.keys(newKeys).length > 0) {
      setExpandedRepoGroups((prev) => ({ ...newKeys, ...prev }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [features]);

  const toggleRepoGroup = (repo: string) => {
    setExpandedRepoGroups((prev) => ({ ...prev, [repo]: !prev[repo] }));
  };

  const groupedConcepts = useMemo(() => {
    const order: string[] = [];
    const map: Record<string, FeatureSummary[]> = {};
    for (const f of features) {
      const repo = f.repo || getRepoFromFeatureId(f.id);
      if (!map[repo]) {
        order.push(repo);
        map[repo] = [];
      }
      map[repo].push(f);
    }
    return order.map((repo) => ({ repo, concepts: map[repo] }));
  }, [features]);

  return (
    <div className="fixed right-0 top-12 bottom-0 w-80 border-l bg-background flex flex-col">
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Docs Section */}
        <div>
          <Button
            variant="ghost"
            className="w-full justify-between p-2 h-auto"
            onClick={() => setIsDocsExpanded(!isDocsExpanded)}
          >
            <div className="flex items-center gap-2">
              <BookOpen className="h-4 w-4" />
              <span className="font-medium">Docs</span>
              <Badge variant="secondary" className="ml-1">
                {docs.length}
              </Badge>
            </div>
            <ChevronDown
              className={cn(
                "h-4 w-4 transition-transform",
                isDocsExpanded && "rotate-180"
              )}
            />
          </Button>

          <AnimatePresence>
            {isDocsExpanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="mt-2 space-y-1">
                  {isDocsLoading ? (
                    <div className="space-y-2 p-2">
                      {[1, 2, 3].map((i) => (
                        <div
                          key={i}
                          className="h-8 bg-muted/30 rounded animate-pulse"
                        />
                      ))}
                    </div>
                  ) : docs.length === 0 ? (
                    <div className="p-4 text-sm text-muted-foreground text-center">
                      No documentation available
                    </div>
                  ) : (
                    docs.map((doc) => {
                      const itemKey = `doc-${doc.repoName}`;
                      const isActive = activeItemKey === itemKey;
                      return (
                        <button
                          key={doc.repoName}
                          onClick={() =>
                            onDocClick(doc.repoName, doc.documentation)
                          }
                          className={cn(
                            "w-full text-left p-2 rounded-md text-sm transition-colors",
                            isActive
                              ? "bg-muted/60 font-medium"
                              : "bg-muted/30 hover:bg-muted/50"
                          )}
                        >
                          {doc.repoName}
                        </button>
                      );
                    })
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Concepts Section */}
        <div>
          <Button
            variant="ghost"
            className="w-full justify-between p-2 h-auto"
            onClick={() => setIsConceptsExpanded(!isConceptsExpanded)}
          >
            <div className="flex items-center gap-2">
              <Lightbulb className="h-4 w-4" />
              <span className="font-medium">Concepts</span>
              <Badge variant="secondary" className="ml-1">
                {features.length}
              </Badge>
            </div>
            <ChevronDown
              className={cn(
                "h-4 w-4 transition-transform",
                isConceptsExpanded && "rotate-180"
              )}
            />
          </Button>

          <AnimatePresence>
            {isConceptsExpanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="mt-2 space-y-1">
                  {isConceptsLoading ? (
                    <div className="space-y-2 p-2">
                      {[1, 2, 3].map((i) => (
                        <div
                          key={i}
                          className="h-8 bg-muted/30 rounded animate-pulse"
                        />
                      ))}
                    </div>
                  ) : features.length === 0 ? (
                    <div className="p-4 text-sm text-muted-foreground text-center">
                      No concepts discovered yet
                    </div>
                  ) : (
                    groupedConcepts.map(({ repo, concepts: group }) => {
                      const shortName = repo.split("/")[1] ?? repo;
                      const isGroupExpanded =
                        expandedRepoGroups[repo] ?? true;
                      return (
                        <div key={repo}>
                          <Button
                            variant="ghost"
                            className="w-full justify-between pl-4 pr-2 py-1 h-auto text-xs text-muted-foreground"
                            onClick={() => toggleRepoGroup(repo)}
                          >
                            <div className="flex items-center gap-1.5">
                              <span className="font-medium">{shortName}</span>
                              <Badge variant="secondary">
                                {group.length}
                              </Badge>
                            </div>
                            <ChevronDown
                              className={cn(
                                "h-3 w-3 transition-transform",
                                isGroupExpanded && "rotate-180"
                              )}
                            />
                          </Button>
                          <AnimatePresence>
                            {isGroupExpanded && (
                              <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: "auto", opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                transition={{ duration: 0.2 }}
                                className="overflow-hidden"
                              >
                                <div className="mt-1 space-y-1 pl-2">
                                  {group.map((concept) => {
                                    const itemKey = `concept-${concept.id}`;
                                    const isActive =
                                      activeItemKey === itemKey;
                                    return (
                                      <button
                                        key={concept.id}
                                        onClick={() =>
                                          onConceptClick(
                                            concept.id,
                                            concept.name,
                                            concept.description
                                          )
                                        }
                                        className={cn(
                                          "w-full text-left p-2 rounded-md text-sm transition-colors",
                                          isActive
                                            ? "bg-muted/60 font-medium"
                                            : "bg-muted/30 hover:bg-muted/50"
                                        )}
                                      >
                                        {concept.name}
                                      </button>
                                    );
                                  })}
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </div>
                      );
                    })
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
