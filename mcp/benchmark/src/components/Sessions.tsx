import { useSessionsState } from "../hooks/useSessionsState";
import { SessionSidebar } from "./sessions/SessionSidebar";
import { SessionDetail } from "./sessions/SessionDetail";

export function Sessions() {
  const state = useSessionsState();
  return (
    <div style={{ display: "flex", gap: "16px", flex: 1, minHeight: 0 }}>
      <SessionSidebar
        loading={state.loading}
        runs={state.runs}
        filteredRuns={state.filteredRuns}
        selected={state.selected}
        quickSearch={state.quickSearch}
        repoSearch={state.repoSearch}
        sourceFilter={state.sourceFilter}
        rangeFilter={state.rangeFilter}
        dayFilter={state.dayFilter}
        repoOptions={state.repoOptions}
        sourceOptions={state.sourceOptions}
        load={state.load}
        loadDetail={state.loadDetail}
        setQuickSearch={state.setQuickSearch}
        setRepoSearch={state.setRepoSearch}
        setSourceFilter={state.setSourceFilter}
        setRangeFilter={state.setRangeFilter}
        setDayFilter={state.setDayFilter}
        clearFilters={state.clearFilters}
      />
      <SessionDetail
        selected={state.selected}
        parsed={state.parsed}
        annotations={state.annotations}
        diagnostics={state.diagnostics}
        flagsById={state.flagsById}
        freq={state.freq}
        selectedUsage={state.selectedUsage}
        prompt={state.prompt}
        answer={state.answer}
        openTurnId={state.openTurnId}
        handleTurnToggle={state.handleTurnToggle}
        handleAnnotate={state.handleAnnotate}
        showSessionAnnotationForm={state.showSessionAnnotationForm}
        setShowSessionAnnotationForm={state.setShowSessionAnnotationForm}
      />
    </div>
  );
}
