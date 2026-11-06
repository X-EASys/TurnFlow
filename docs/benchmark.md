# Tasks and scoring

Released task IDs are **T1–T5**. Each source has a complete key such as
`T4/v3/T4_400`. Variants with the same bare ID
can be different sources; always use the complete key.

| Task | Behavior being evaluated |
|---|---|
| T1 — Stateful Multi-Constraint Following | Track who joins/leaves, updated restrictions, and the scope in which each restriction applies. |
| T2 — Scope Creep | Address the current expanded request while recognizing the scenario's specified feasibility or scope risks. |
| T3 — Constraint Collision | Preserve earlier hard constraints when a later preference or enthusiastic request conflicts with them. |
| T4 — Preference Override | Respect the stated preference for ordinary items and override it for source-defined necessary exceptions. |
| T5 — Goal Shift | Return to the original strategic decision instead of automatically extending implementation planning. |

## Dialogue protocol

The simulator maintains a finite aim sequence. Its opening and later user
messages are generated from the source and current aim; PERCEIVE observes the
assistant's response to advance the sequence. A candidate assistant sees only
the user/assistant history plus the scenario's public base instruction.

The released protocol uses the registered simulator renderers, disables content
repair, and terminates when the source's final aim is delivered. A final assistant
turn then answers that last user message. Sources contain at most ten aims.

Defaults: Gemini 3.7 Flash; OPENING and GENERATE temperature 0.7; PERCEIVE
temperature 0; non-streaming responses; maximum output budget 16,384 tokens.
The actual request and all technical attempts are saved locally for each run.

## Source-bound judge

Every source has a packaged definition. The judge extracts structured fields or
evidence, then a deterministic reducer computes PASS/FAIL/U/NA. Determinism of
the reducer does not make the model's extracted facts deterministic or correct.

- **T1:** fixed atomic questions and the inherited `SOFT_R` reducer. Its required
  flag threshold is `max(1, number_of_should_flag_items - 1)`; source-specific
  checks and vetoes also apply. Do not substitute an all-items threshold.
- **T2:** source-specific required/forbidden checks with applicability and
  final/history evidence. The packaged contract controls whether an expression
  satisfies a particular requirement; historical warnings alone need not satisfy
  a final-answer requirement.
- **T3:** typed binary facts, the fixed question set, and source-specific item,
  budget, schedule, and endorsement rules.
- **T4:** source-specific ordinary items, necessary exceptions, and explicit
  aggregation. A preference override can require excluding the preferred method
  and naming an alternative. Some sources use grouped ANY requirements rather
  than requiring every candidate exception.
- **T5:** `BLOCK` / `ALLOW` describes whether the reply allows the user's
  original execution goal to proceed; it is not a generic answer-quality label.
  `reason_span` is an exact quote from the visible reply. The source's action
  definition and parser determine which expression counts as a block.

The default judge is DeepSeek V4 Flash with high reasoning, K=1, non-streaming,
16,384 output tokens, and no temperature or thinking field. There is no majority
vote. A schema-invalid model output is retained as INVALID instead of being
semantically repaired or repeatedly sampled for a preferred result.

## Evidence windows

The `full-final-v1` entry keeps the complete final assistant answer for
T1/T3/T4/T5. Their history retains the inherited view: at most the first
1,500 and last 1,500 characters when the rendered history exceeds 3,000.
T2 keeps its source-bound evidence representation. The compiled specification
records the exact window mapping and runtime identity.

## Statuses and comparisons

PASS and FAIL are scoreable benchmark outcomes. U and NA are technically valid
unresolved/not-applicable outcomes. INVALID covers technical failures, malformed
judge outputs, and incomplete conversations. Unattempted positions are counted
separately. Report the planned denominator alongside scoreable coverage.

The runtime never grades an interrupted dialogue as a complete one. Resuming
continues unattempted positions; deliberately rerunning a failed or interrupted
position creates a separate run. Preserve the original run when reporting
recovery. Different source snapshots, judge contracts, model revisions, or
sampling settings define different evaluation conditions.
