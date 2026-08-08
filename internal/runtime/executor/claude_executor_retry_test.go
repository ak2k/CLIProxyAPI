package executor

import (
	"net/http"
	"strconv"
	"testing"
	"time"
)

func epochIn(d time.Duration) string {
	return strconv.FormatInt(time.Now().Add(d).Unix(), 10)
}

// TestNewClaudeStatusErrPopulatesRetryAfterOnErrorStatuses asserts the status
// gate in newClaudeStatusErr. The gate spans the whole 4xx/5xx range, not just
// 429, so a Retry-After on a 503 is honoured too; a success status carries no
// cooldown. Header-parsing correctness is covered by the helps package; this
// test only verifies the wrapping logic that lives alongside the executor.
func TestNewClaudeStatusErrPopulatesRetryAfterOnErrorStatuses(t *testing.T) {
	headers := http.Header{
		"Retry-After": {"60"},
	}

	for _, status := range []int{429, 500, 401} {
		if got := newClaudeStatusErr(status, headers, []byte("error")); got.retryAfter == nil {
			t.Fatalf("%d: expected retryAfter to be set", status)
		}
	}
	if got := newClaudeStatusErr(200, headers, []byte("ok")); got.retryAfter != nil {
		t.Fatalf("200: expected retryAfter to be nil, got %v", *got.retryAfter)
	}
}

// TestNewClaudeStatusErrCooldownValues pins which parser answers for each shape
// of rate-limit header, by value rather than by presence. Deleting the fallback
// branch or swapping the two parsers' precedence changes at least one of these
// numbers, which a nil/non-nil assertion would not catch.
//
// Where the upstream parser answers, the expected value is a range: it adds a
// bounded random grace of 1–30s to every cooldown it returns.
func TestNewClaudeStatusErrCooldownValues(t *testing.T) {
	const fuzz = 31 * time.Second

	tests := []struct {
		name    string
		headers http.Header
		wantNil bool
		wantLow time.Duration
		wantHi  time.Duration
	}{
		{
			// The regression this gate exists for. An ordinary model-level 429
			// carries a routine unified-reset next to healthy windows; honouring
			// it would park the credential for hours on a response that is
			// classified model-scoped.
			name: "healthy windows with a routine unified-reset yield no cooldown",
			headers: http.Header{
				"Anthropic-Ratelimit-Unified-5h-Status": {"allowed"},
				"Anthropic-Ratelimit-Unified-7d-Status": {"allowed"},
				"Anthropic-Ratelimit-Unified-Reset":     {epochIn(4 * time.Hour)},
			},
			wantNil: true,
		},
		{
			// A genuine rejection: upstream's parser owns window selection and
			// answers first. If precedence swapped, the Date-less fallback would
			// return a bare 4h with no fuzz and the low bound would fail.
			name: "rejected 5h window uses the upstream parser's window selection",
			headers: http.Header{
				"Anthropic-Ratelimit-Unified-5h-Status": {"rejected"},
				"Anthropic-Ratelimit-Unified-5h-Reset":  {epochIn(4 * time.Hour)},
			},
			// The low bound carries a second of slack: the reset epoch is
			// truncated to whole seconds, so the smallest fuzz draw can land
			// fractionally under 4h.
			wantLow: 4*time.Hour - time.Second,
			wantHi:  4*time.Hour + fuzz,
		},
		{
			// The fallback's one unique contribution. The client clock runs an
			// hour ahead of the server, so upstream reads the reset as already
			// past and declines; anchoring to the response's own Date recovers
			// the true 5 minutes. Without the fallback this is nil.
			name: "clock skew on a rejection is rescued by anchoring to Date",
			headers: http.Header{
				"Anthropic-Ratelimit-Unified-Status": {"rejected"},
				"Date":                               {time.Now().Add(-time.Hour).UTC().Format(http.TimeFormat)},
				"Anthropic-Ratelimit-Unified-Reset":  {epochIn(-55 * time.Minute)},
			},
			wantLow: 5*time.Minute - 2*time.Second,
			wantHi:  5*time.Minute + 2*time.Second,
		},
		{
			// Side effect of the gate, pinned deliberately: with no Anthropic
			// headers there is no declared rejection, so the fallback is never
			// consulted and a zero-length cooldown is not manufactured. The
			// generic exponential backoff takes it from here.
			name: "Retry-After zero alone yields no cooldown",
			headers: http.Header{
				"Retry-After": {"0"},
			},
			wantNil: true,
		},
		{
			// Upstream's parser handles a bare Retry-After on its own; the
			// fallback is not needed and not reached.
			name: "bare Retry-After is answered by the upstream parser",
			headers: http.Header{
				"Retry-After": {"60"},
			},
			wantLow: 60 * time.Second,
			wantHi:  60*time.Second + fuzz,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := newClaudeStatusErr(http.StatusTooManyRequests, tc.headers, []byte("rate limited")).retryAfter
			if tc.wantNil {
				if got != nil {
					t.Fatalf("want no cooldown, got %v", *got)
				}
				return
			}
			if got == nil {
				t.Fatalf("want cooldown in [%v, %v], got none", tc.wantLow, tc.wantHi)
			}
			if *got < tc.wantLow || *got > tc.wantHi {
				t.Fatalf("want cooldown in [%v, %v], got %v", tc.wantLow, tc.wantHi, *got)
			}
		})
	}
}

// TestNewClaudeStatusErrCapsFarFutureReset pins the cap that the upstream parser
// has no equivalent of. A malformed reset — the shape an off-by-1000 ms-vs-s
// epoch takes — must never park a credential beyond the longest window
// Anthropic actually enforces.
//
// Both inputs reach the cap by different routes. The representable one is
// capped directly. The saturating one overflows in the upstream parser, whose
// Sub pins at the maximum Duration and whose fuzz then wraps it negative; the
// fallback re-parses it and the cap applies to that.
func TestNewClaudeStatusErrCapsFarFutureReset(t *testing.T) {
	tests := []struct {
		name  string
		reset string
	}{
		{
			name:  "representable far-future reset",
			reset: epochIn(300 * 24 * time.Hour),
		},
		{
			name:  "reset that saturates time.Duration",
			reset: "99999999999",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			headers := http.Header{
				"Anthropic-Ratelimit-Unified-Status": {"rejected"},
				"Anthropic-Ratelimit-Unified-Reset":  {tc.reset},
			}

			got := newClaudeStatusErr(http.StatusTooManyRequests, headers, []byte("rate limited")).retryAfter
			if got == nil {
				t.Fatal("expected a capped cooldown, got none")
			}
			if *got != 7*24*time.Hour {
				t.Fatalf("want cooldown capped at 7d, got %v", *got)
			}
		})
	}
}

// TestNewClaudeFastDirectResponseErrorCapsCooldown covers the fast-mode 429,
// which reaches the cooldown path through its own error type rather than
// through newClaudeStatusErr. It is credential-scoped, so an uncapped reset
// parks the whole credential rather than one model.
//
// Both inputs are the far-future resets the cap exists for, and both must come
// back at exactly the cap — including the saturating one, which only lands
// there because this path derives its cooldown the same way the ordinary one
// does and so inherits the fallback that re-parses an overflowed duration.
func TestNewClaudeFastDirectResponseErrorCapsCooldown(t *testing.T) {
	tests := []struct {
		name  string
		reset string
	}{
		{
			name:  "representable far-future reset",
			reset: epochIn(300 * 24 * time.Hour),
		},
		{
			name:  "reset that saturates time.Duration",
			reset: "99999999999",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			resp := &http.Response{
				StatusCode: http.StatusTooManyRequests,
				Header: http.Header{
					"Anthropic-Ratelimit-Unified-Status": {"rejected"},
					"Anthropic-Ratelimit-Unified-Reset":  {tc.reset},
				},
			}

			err := newClaudeFastDirectResponseError(resp, []byte(`{"error":"rate limited"}`))
			rap, ok := err.(interface{ RetryAfter() *time.Duration })
			if !ok {
				t.Fatal("fast direct response error does not expose RetryAfter")
			}

			got := rap.RetryAfter()
			if got == nil {
				t.Fatal("expected a capped cooldown, got none")
			}
			if *got != 7*24*time.Hour {
				t.Fatalf("want cooldown capped at 7d, got %v", *got)
			}
		})
	}
}
