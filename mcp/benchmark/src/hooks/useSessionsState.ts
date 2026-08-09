import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { api } from "../api";
import {
  buildToolFrequency,
  usageOf,
} from "../utils";
import { parseTrace } from "../trace/parse";
import { analyzeTrace } from "../trace/analyze";
import type { ParsedTrace, TraceAnalysis, IssueKind } from "../trace/types";
import type { ProductionRun, TokenUsage } from "../types";
import type { Annotation, AnnotationMarker } from "../components/Annotations";

export const PAGE_LIMIT = 100;

export interface SessionsState {
  loading: boolean;
  loadingMore: boolean;
  hasMore: boolean;
  runs: ProductionRun[];
  selected: ProductionRun | null;
  annotations: Annotation[];
  repoSearch: string;
  sourceFilter: string;
  rangeFilter: "24h" | "7d" | "30d" | "3m" | "1y" | "all";
  dayFilter: string;
  repoOptions: string[];
  sourceOptions: string[];
  openTurnId: string | null;
  parsed: ParsedTrace;
  diagnostics: TraceAnalysis;
  flagsById: Map<string, IssueKind[]>;
  freq: Array<{ toolName: string; count: number }>;
  selectedUsage: TokenUsage | null;
  prompt: string;
  answer: string;
  showSessionAnnotationForm: boolean;
  load: () => void;
  loadMore: () => void;
  loadDetail: (run: ProductionRun) => void;
  handleAnnotate: (
    marker: AnnotationMarker,
    note: string,
    toolCallId?: string,
  ) => void;
  handleTurnToggle: (turnId: string) => void;
  setRepoSearch: (v: string) => void;
  setSourceFilter: (v: string) => void;
  setRangeFilter: (v: "24h" | "7d" | "30d" | "3m" | "1y" | "all") => void;
  setDayFilter: (v: string) => void;
  clearFilters: () => void;
  setShowSessionAnnotationForm: (v: boolean) => void;
}

const EMPTY_PARSED: ParsedTrace = {
  userPrompt: "",
  answer: "",
  calls: [],
  results: [],
  events: [],
  turns: [],
};

export function useSessionsState(): SessionsState {
  const [searchParams, setSearchParams] = useSearchParams();
  const [runs, setRuns] = useState<ProductionRun[]>([]);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [selected, setSelected] = useState<ProductionRun | null>(null);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [showSessionAnnotationForm, setShowSessionAnnotationForm] =
    useState(false);
  const [loading, setLoading] = useState(true);
  const [repoSearch, setRepoSearchRaw] = useState(
    searchParams.get("repo") || "",
  );
  const [debouncedRepo, setDebouncedRepo] = useState(repoSearch);
  const [sourceFilter, setSourceFilter] = useState(
    searchParams.get("source") || "all",
  );
  const [rangeFilter, setRangeFilter] = useState<
    "24h" | "7d" | "30d" | "3m" | "1y" | "all"
  >((searchParams.get("range") as "24h" | "7d" | "30d" | "3m" | "1y" | "all") || "all");
  const [dayFilter, setDayFilter] = useState(searchParams.get("day") || "");
  const [openTurnId, setOpenTurnId] = useState<string | null>(null);

  // Facets (full-dataset repo/source options)
  const [repoOptions, setRepoOptions] = useState<string[]>([]);
  const [sourceOptions, setSourceOptions] = useState<string[]>([]);

  // Request generation counter — incremented on filter reset to cancel
  // in-flight loadMore calls that would otherwise append stale pages.
  const genRef = useRef(0);

  // Debounce repo search input (300ms) to avoid a request per keystroke
  const repoDebounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const setRepoSearch = useCallback((v: string) => {
    setRepoSearchRaw(v);
    if (repoDebounceTimer.current) clearTimeout(repoDebounceTimer.current);
    repoDebounceTimer.current = setTimeout(() => setDebouncedRepo(v), 300);
  }, []);

  // Fetch facets once on mount
  useEffect(() => {
    api.sessions
      .facets()
      .then(({ repos, sources }) => {
        setRepoOptions(repos.filter(Boolean));
        setSourceOptions(sources.filter(Boolean));
      })
      .catch(() => {
        // non-fatal — dropdowns will be empty
      });
  }, []);

  /**
   * Load the first page (offset=0), replacing the `runs` array.
   * Called on mount and whenever a filter changes.
   */
  const load = useCallback(async () => {
    const gen = ++genRef.current;
    setLoading(true);
    setOffset(0);
    setHasMore(true);
    try {
      const page = await api.sessions.list({
        limit: PAGE_LIMIT,
        offset: 0,
        source: sourceFilter !== "all" ? sourceFilter : undefined,
        repo: debouncedRepo || undefined,
        range: rangeFilter !== "all" ? rangeFilter : undefined,
        day: dayFilter || undefined,
      });
      if (gen !== genRef.current) return; // stale
      setRuns(page);
      setHasMore(page.length === PAGE_LIMIT);
      setOffset(page.length);
    } catch (e: any) {
      if (gen !== genRef.current) return;
      toast.error(e.message);
    } finally {
      if (gen === genRef.current) setLoading(false);
    }
  }, [sourceFilter, debouncedRepo, rangeFilter, dayFilter]);

  /**
   * Append the next page. Called by the IntersectionObserver sentinel.
   * Uses a generation counter to discard responses from stale requests.
   */
  const loadMore = useCallback(async () => {
    if (loadingMore || !hasMore) return;
    const gen = genRef.current;
    setLoadingMore(true);
    try {
      const currentOffset = offset;
      const page = await api.sessions.list({
        limit: PAGE_LIMIT,
        offset: currentOffset,
        source: sourceFilter !== "all" ? sourceFilter : undefined,
        repo: debouncedRepo || undefined,
        range: rangeFilter !== "all" ? rangeFilter : undefined,
        day: dayFilter || undefined,
      });
      if (gen !== genRef.current) return; // filter changed mid-flight
      setRuns((prev) => {
        // De-dup by id in case new sessions arrived between pages
        const seen = new Set(prev.map((r) => r.id));
        const fresh = page.filter((r) => !seen.has(r.id));
        return [...prev, ...fresh];
      });
      setHasMore(page.length === PAGE_LIMIT);
      setOffset((prev) => prev + page.length);
    } catch (e: any) {
      if (gen !== genRef.current) return;
      toast.error(e.message);
    } finally {
      if (gen === genRef.current) setLoadingMore(false);
    }
  }, [loadingMore, hasMore, offset, sourceFilter, debouncedRepo, rangeFilter, dayFilter]);

  // Reload page-1 whenever filters change
  useEffect(() => {
    load();
  }, [load]);

  const loadDetail = async (run: ProductionRun) => {
    try {
      const detail = await api.sessions.get(run.id);
      setSelected(detail);
      setAnnotations(detail.annotations ?? []);
      setShowSessionAnnotationForm(false);
    } catch {
      setSelected(run);
      setAnnotations(run.annotations ?? []);
      setShowSessionAnnotationForm(false);
    }
  };

  const handleAnnotate = useCallback(
    async (marker: AnnotationMarker, note: string, toolCallId?: string) => {
      if (!selected) return;
      try {
        const ann = await api.sessions.annotate(selected.id, {
          target: toolCallId ? "tool_call" : "session",
          target_id: toolCallId,
          marker,
          note: note || undefined,
        });
        setAnnotations((prev) => [...prev, ann]);
        toast.success("Annotation saved");
      } catch (e: any) {
        toast.error(e.message);
      }
    },
    [selected],
  );

  const handleTurnToggle = useCallback((turnId: string) => {
    setOpenTurnId((prev) => (prev === turnId ? null : turnId));
  }, []);

  const freq = selected ? buildToolFrequency(selected.tool_sequence) : [];

  const parsed = useMemo(
    () => (selected ? parseTrace(selected.trace) : EMPTY_PARSED),
    [selected],
  );

  const selectedUsage = selected ? usageOf(selected.token_usage) : null;

  const prompt =
    parsed.userPrompt || selected?.user_prompt_preview || "No prompt preview";
  const answer =
    parsed.answer || selected?.answer_preview || "No answer preview";

  const diagnostics = useMemo(
    () => analyzeTrace(parsed, prompt),
    [parsed, prompt],
  );

  const flagsById = useMemo(
    () => new Map(diagnostics.steps.map((s) => [s.id, s.flags])),
    [diagnostics.steps],
  );

  // Sync URL search params ↔ filter state
  useEffect(() => {
    setRepoSearchRaw(searchParams.get("repo") || "");
    setDebouncedRepo(searchParams.get("repo") || "");
    setSourceFilter(searchParams.get("source") || "all");
    setRangeFilter(
      (searchParams.get("range") as "24h" | "7d" | "30d" | "3m" | "1y" | "all") || "all",
    );
    setDayFilter(searchParams.get("day") || "");
  }, [searchParams]);

  useEffect(() => {
    const nextParams = new URLSearchParams();
    if (repoSearch) nextParams.set("repo", repoSearch);
    if (sourceFilter !== "all") nextParams.set("source", sourceFilter);
    if (rangeFilter !== "all") nextParams.set("range", rangeFilter);
    if (dayFilter) nextParams.set("day", dayFilter);
    setSearchParams(nextParams, { replace: true });
  }, [dayFilter, repoSearch, rangeFilter, setSearchParams, sourceFilter]);

  useEffect(() => {
    if (parsed.turns.length === 0) {
      setOpenTurnId(null);
      return;
    }
    const preferredTurn =
      parsed.turns.find((turn) => turn.kind !== "setup") ?? parsed.turns[0];
    setOpenTurnId(preferredTurn.id);
  }, [selected?.id, parsed.turns]);

  const clearFilters = () => {
    setRepoSearch("");
    setSourceFilter("all");
    setRangeFilter("all");
    setDayFilter("");
  };

  return {
    loading,
    loadingMore,
    hasMore,
    runs,
    selected,
    annotations,
    repoSearch,
    sourceFilter,
    rangeFilter,
    dayFilter,
    repoOptions,
    sourceOptions,
    openTurnId,
    parsed,
    diagnostics,
    flagsById,
    freq,
    selectedUsage,
    prompt,
    answer,
    showSessionAnnotationForm,
    load,
    loadMore,
    loadDetail,
    handleAnnotate,
    handleTurnToggle,
    setRepoSearch,
    setSourceFilter,
    setRangeFilter,
    setDayFilter,
    clearFilters,
    setShowSessionAnnotationForm,
  };
}
