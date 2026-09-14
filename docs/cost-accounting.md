# Cost accounting

`ModelUsage` records each dispatched provider attempt in `TaskStore.model_usage`, including retries, failures, compaction, embeddings, transcription and dreaming. Async-local scope carries task, requester and background-job ownership. Calls cancelled while waiting for capacity do not dispatch and do not produce a usage attempt.

The record captures the model, purpose, status, duration, reported input/output tokens and the estimate computed from that profile's prices at the time. Missing data is null. Embedding attempts currently lack complete input/output pricing in this estimator and remain unpriced. Cache discounts, provider web-service fees and invoice adjustments are not comprehensively represented. This is operational accounting, not a billing ledger.

`readCostReport` aggregates by actor and rolling date range, with a separate operator-only installation scope. It reads no prompt or task content. Indexes support actor/date and installation/date queries. The panel shows known cost separately from counts of unknown cost and incomplete token data. Failed requests with no usage are not free. Model rows are limited to the top eight by known cost, while totals include all models.

`/costs` opens the actor's report in the channel by default; `ephemeral:true` makes it private. `/settings` has a Custos tab that updates the settings message with the operator report. Button IDs bind to the viewer; each installation report rechecks operator authority. User reports cannot request another actor through command arguments or component IDs.

Task forgetting removes associated attempt records. Unowned maintenance remains installation-only. Pre-v5 aggregates are not mixed with per-attempt rows, which would double count some work; reports describe retained v5 attempt records. There is no budget enforcement, spending threshold, or stop condition in this service.

Tests cover actor isolation, exact date boundaries, future-date exclusion, retries, failed/unknown usage, unowned background rows, empty reports, foreign controls, revoked operators and bounded component content.
