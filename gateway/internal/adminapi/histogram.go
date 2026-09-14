package adminapi

import (
	"math"
	"net/http"
	"sort"
	"time"
)

// Phase-7 token and latency histograms. Same shape family as
// /_plugin/histogram/cost (observability.go): ?window=, ?bucket=,
// ?dimension=, optional ?user_id= / ?agent_name= scoping, epoch-
// aligned buckets, empty buckets omitted, series sorted heaviest
// first. Bifrost's own /api/logs/histogram/*/by-dimension can't group
// by metadata columns, so these bucket in Go like cost does.

// TokenHistogramPoint is one bucket of a per-dimension token series.
type TokenHistogramPoint struct {
	Timestamp        string `json:"ts"`
	PromptTokens     int64  `json:"prompt_tokens"`
	CompletionTokens int64  `json:"completion_tokens"`
	TotalTokens      int64  `json:"total_tokens"`
}

// TokenHistogramSeries is one dimension value's line.
type TokenHistogramSeries struct {
	DimensionValue string                `json:"dimension_value"`
	Points         []TokenHistogramPoint `json:"points"`
}

// HistogramTokensResponse is the envelope for /_plugin/histogram/tokens.
type HistogramTokensResponse struct {
	BucketSizeSeconds int64                  `json:"bucket_size_seconds"`
	Dimension         string                 `json:"dimension"`
	Series            []TokenHistogramSeries `json:"series"`
}

// LatencyHistogramPoint is one bucket of a per-dimension latency
// series: nearest-rank percentiles (ms) over the bucket's calls plus
// the call count the percentiles were taken from, so a p99 over
// three calls can be read with the right skepticism.
type LatencyHistogramPoint struct {
	Timestamp string  `json:"ts"`
	P50       float64 `json:"p50"`
	P95       float64 `json:"p95"`
	P99       float64 `json:"p99"`
	Count     int64   `json:"count"`
}

// LatencyHistogramSeries is one dimension value's line.
type LatencyHistogramSeries struct {
	DimensionValue string                  `json:"dimension_value"`
	Points         []LatencyHistogramPoint `json:"points"`
}

// HistogramLatencyResponse is the envelope for /_plugin/histogram/latency.
type HistogramLatencyResponse struct {
	BucketSizeSeconds int64                    `json:"bucket_size_seconds"`
	Dimension         string                   `json:"dimension"`
	Series            []LatencyHistogramSeries `json:"series"`
}

// histogramArgs parses the three histogram params in the order the
// cost handler does (window → bucket → dimension), so every histogram
// rejects bad input with the same messages. When ok is false a 400
// has been written.
func histogramArgs(w http.ResponseWriter, r *http.Request) (start, end time.Time, bucket time.Duration, dimension string, ok bool) {
	_, start, end, ok = parseWindow(w, r)
	if !ok {
		return
	}
	bucket, ok = parseBucket(w, r, end.Sub(start))
	if !ok {
		return
	}
	dimension, ok = parseDimensionParam(w, r)
	return
}

// bucketStart floors a row's timestamp to its epoch-aligned bucket.
// Rows with an unparseable timestamp are skipped (ok=false) rather
// than piled into bucket zero.
func bucketStart(ts string, bucketSec int64) (int64, bool) {
	t, err := time.Parse(time.RFC3339Nano, ts)
	if err != nil {
		return 0, false
	}
	return (t.Unix() / bucketSec) * bucketSec, true
}

func bucketLabel(start int64) string {
	return time.Unix(start, 0).UTC().Format(time.RFC3339)
}

// ─── /_plugin/histogram/tokens ───────────────────────────────────────

func (h *observabilityHandlers) histogramTokens(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	start, end, bucket, dimension, ok := histogramArgs(w, r)
	if !ok {
		return
	}
	logs, err := h.logs.searchAll(r.Context(), searchOpts{
		StartTime: &start,
		EndTime:   &end,
		Metadata:  metadataFilterFromQuery(r),
	}, 1000, 200_000)
	if err != nil {
		writeUpstreamError(w, err, "histogram.tokens")
		return
	}

	bucketSec := int64(bucket.Seconds())
	type agg struct{ prompt, completion, total int64 }
	series := map[string]map[int64]*agg{}
	for _, l := range logs {
		dim := dimensionValue(l, dimension)
		if dim == "" || l.TokenUsage == nil {
			continue
		}
		b, ok := bucketStart(l.Timestamp, bucketSec)
		if !ok {
			continue
		}
		if _, ok := series[dim]; !ok {
			series[dim] = map[int64]*agg{}
		}
		a, ok := series[dim][b]
		if !ok {
			a = &agg{}
			series[dim][b] = a
		}
		a.prompt += l.TokenUsage.PromptTokens
		a.completion += l.TokenUsage.CompletionTokens
		a.total += l.tokens()
	}

	out := HistogramTokensResponse{
		BucketSizeSeconds: bucketSec,
		Dimension:         dimension,
		Series:            make([]TokenHistogramSeries, 0, len(series)),
	}
	totals := map[string]int64{}
	for dim, buckets := range series {
		pts := make([]TokenHistogramPoint, 0, len(buckets))
		for b, a := range buckets {
			pts = append(pts, TokenHistogramPoint{
				Timestamp:        bucketLabel(b),
				PromptTokens:     a.prompt,
				CompletionTokens: a.completion,
				TotalTokens:      a.total,
			})
			totals[dim] += a.total
		}
		sort.Slice(pts, func(i, j int) bool { return pts[i].Timestamp < pts[j].Timestamp })
		out.Series = append(out.Series, TokenHistogramSeries{DimensionValue: dim, Points: pts})
	}
	sort.SliceStable(out.Series, func(i, j int) bool {
		ti, tj := totals[out.Series[i].DimensionValue], totals[out.Series[j].DimensionValue]
		if ti != tj {
			return ti > tj
		}
		return out.Series[i].DimensionValue < out.Series[j].DimensionValue
	})
	writeJSON(w, http.StatusOK, out)
}

// ─── /_plugin/histogram/latency ──────────────────────────────────────

func (h *observabilityHandlers) histogramLatency(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	start, end, bucket, dimension, ok := histogramArgs(w, r)
	if !ok {
		return
	}
	logs, err := h.logs.searchAll(r.Context(), searchOpts{
		StartTime: &start,
		EndTime:   &end,
		Metadata:  metadataFilterFromQuery(r),
	}, 1000, 200_000)
	if err != nil {
		writeUpstreamError(w, err, "histogram.latency")
		return
	}

	bucketSec := int64(bucket.Seconds())
	series := map[string]map[int64][]float64{}
	for _, l := range logs {
		dim := dimensionValue(l, dimension)
		if dim == "" || l.Latency <= 0 {
			continue // no latency on errored / in-flight rows
		}
		b, ok := bucketStart(l.Timestamp, bucketSec)
		if !ok {
			continue
		}
		if _, ok := series[dim]; !ok {
			series[dim] = map[int64][]float64{}
		}
		series[dim][b] = append(series[dim][b], l.Latency)
	}

	out := HistogramLatencyResponse{
		BucketSizeSeconds: bucketSec,
		Dimension:         dimension,
		Series:            make([]LatencyHistogramSeries, 0, len(series)),
	}
	counts := map[string]int64{}
	for dim, buckets := range series {
		pts := make([]LatencyHistogramPoint, 0, len(buckets))
		for b, lat := range buckets {
			sort.Float64s(lat)
			pts = append(pts, LatencyHistogramPoint{
				Timestamp: bucketLabel(b),
				P50:       percentile(lat, 0.50),
				P95:       percentile(lat, 0.95),
				P99:       percentile(lat, 0.99),
				Count:     int64(len(lat)),
			})
			counts[dim] += int64(len(lat))
		}
		sort.Slice(pts, func(i, j int) bool { return pts[i].Timestamp < pts[j].Timestamp })
		out.Series = append(out.Series, LatencyHistogramSeries{DimensionValue: dim, Points: pts})
	}
	sort.SliceStable(out.Series, func(i, j int) bool {
		ci, cj := counts[out.Series[i].DimensionValue], counts[out.Series[j].DimensionValue]
		if ci != cj {
			return ci > cj
		}
		return out.Series[i].DimensionValue < out.Series[j].DimensionValue
	})
	writeJSON(w, http.StatusOK, out)
}

// percentile is the nearest-rank percentile of an ascending-sorted
// sample: the smallest value with at least p of the sample at or
// below it. No interpolation — with the small per-bucket counts a
// dashboard sees, a real observed latency is more honest than a
// synthetic one between two.
func percentile(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	rank := int(math.Ceil(p*float64(n))) - 1
	if rank < 0 {
		rank = 0
	}
	if rank >= n {
		rank = n - 1
	}
	return sorted[rank]
}
