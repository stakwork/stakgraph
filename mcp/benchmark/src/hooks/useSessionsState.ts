import { useState, useEffect, useCallback, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { api } from "../api";
import {
  buildToolFrequency,
  usageOf,
  getRangeStart,
} from "../utils";
import { parseTrace } from "../trace/parse";
import { analyzeTrace } from "../trace/analyze";
import type { ParsedTrace, TraceAnalysis, IssueKind } from "../trace/types";
import type { ProductionRun, TokenUsage } from "../types";
import type { Annotation, AnnotationMarker } from "../components/Annotations";

export interface SessionsState {
  loading: boolean;
  runs: ProductionRun[];
  filteredRuns: ProductionRun[];
  selected: ProductionRun | null;
  annotations: Annotation[];
  quickSearch: string;
  repoSearch: string;
  sourceFilter: string;
  rangeFilter: "24h" | "7d" | "30d" | "all";
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
  loadDetail: (run: ProductionRun) => void;
  handleAnnotate: (
    marker: AnnotationMarker,
    note: string,
    toolCallId?: string,
  ) => void;
  handleTurnToggle: (turnId: string) => void;
  setQuickSearch: (v: string) => void;
  setRepoSearch: (v: string) => void;
  setSourceFilter: (v: string) => void;
  setRangeFilter: (v: "24h" | "7d" | "30d" | "all") => void;
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
  const [selected, setSelected] = useState<ProductionRun | null>(null);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [showSessionAnnotationForm, setShowSessionAnnotationForm] =
    useState(false);
  const [loading, setLoading] = useState(true);
  const [quickSearch, setQuickSearch] = useState(
    searchParams.get("q") || "",
  );
  const [repoSearch, setRepoSearch] = useState(
    searchParams.get("repo") || "",
  );
  const [sourceFilter, setSourceFilter] = useState(
    searchParams.get("source") || "all",
  );
  const [rangeFilter, setRangeFilter] = useState<
    "24h" | "7d" | "30d" | "all"
  >((searchParams.get("range") as "24h" | "7d" | "30d" | "all") || "all");
  const [dayFilter, setDayFilter] = useState(searchParams.get("day") || "");
  const [openTurnId, setOpenTurnId] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const sessions = await api.sessions.list();
      setRuns(sessions);
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

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

  const repoOptions = useMemo(
    () => [...new Set(runs.map((r) => r.repo))].sort(),
    [runs],
  );

  const sourceOptions = useMemo(
    () => [...new Set(runs.map((r) => r.source || "unknown"))].sort(),
    [runs],
  );

  useEffect(() => {
    setQuickSearch(searchParams.get("q") || "");
    setRepoSearch(searchParams.get("repo") || "");
    setSourceFilter(searchParams.get("source") || "all");
    setRangeFilter(
      (searchParams.get("range") as "24h" | "7d" | "30d" | "all") || "all",
    );
    setDayFilter(searchParams.get("day") || "");
  }, [searchParams]);

  useEffect(() => {
    const nextParams = new URLSearchParams();
    if (quickSearch) nextParams.set("q", quickSearch);
    if (repoSearch) nextParams.set("repo", repoSearch);
    if (sourceFilter !== "all") nextParams.set("source", sourceFilter);
    if (rangeFilter !== "all") nextParams.set("range", rangeFilter);
    if (dayFilter) nextParams.set("day", dayFilter);
    setSearchParams(nextParams, { replace: true });
  }, [dayFilter, quickSearch, repoSearch, rangeFilter, setSearchParams, sourceFilter]);

  useEffect(() => {
    if (parsed.turns.length === 0) {
      setOpenTurnId(null);
      return;
    }
    const preferredTurn =
      parsed.turns.find((turn) => turn.kind !== "setup") ?? parsed.turns[0];
    setOpenTurnId(preferredTurn.id);
  }, [selected?.id, parsed.turns]);

  const filteredRuns = useMemo(
    () =>
      runs.filter((r) => {
        if (sourceFilter !== "all" && (r.source || "unknown") !== sourceFilter)
          return false;
        if (
          repoSearch &&
          !r.repo.toLowerCase().includes(repoSearch.toLowerCase())
        )
          return false;
        if (
          dayFilter &&
          new Date(r.timestamp).toISOString().slice(0, 10) !== dayFilter
        )
          return false;
        const rangeStart = getRangeStart(rangeFilter);
        if (rangeStart && new Date(r.timestamp).getTime() < rangeStart)
          return false;
        if (quickSearch) {
          const q = quickSearch.toLowerCase();
          const matches =
            r.id.toLowerCase().includes(q) ||
            r.repo.toLowerCase().includes(q) ||
            (r.source || "").toLowerCase().includes(q) ||
            (r.model || "").toLowerCase().includes(q) ||
            (r.user_prompt_preview || "").toLowerCase().includes(q);
          if (!matches) return false;
        }
        return true;
      }),
    [runs, dayFilter, quickSearch, repoSearch, rangeFilter, sourceFilter],
  );

  const clearFilters = () => {
    setQuickSearch("");
    setRepoSearch("");
    setSourceFilter("all");
    setRangeFilter("all");
    setDayFilter("");
  };

  return {
    loading,
    runs,
    filteredRuns,
    selected,
    annotations,
    quickSearch,
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
    loadDetail,
    handleAnnotate,
    handleTurnToggle,
    setQuickSearch,
    setRepoSearch,
    setSourceFilter,
    setRangeFilter,
    setDayFilter,
    clearFilters,
    setShowSessionAnnotationForm,
  };
}
