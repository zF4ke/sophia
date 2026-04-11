# Cleanup And Migration Notes

This codebase intentionally removed older prompt-harness and cache-heavy paths that conflicted with the current runtime.

## Removed Or Replaced

- Large inline prompt blobs were replaced by `resources/prompts/`.
- The old Gemini-only AI facade was replaced by the current model gateway and graph runtime.
- Context stuffing and fake web-search behavior were removed from the active runtime.
- The monolithic security and UI services were split into smaller modules.
- Legacy novelty commands and dead helper modules were removed.
- Refusal-first grounding was replaced by best-effort conversational recovery.

## Current Direction

Sophia is conversational first.

Local storage is a Discord retrieval cache plus runtime state. It is not a separate memory-search product.

When evidence is weak, the runtime should keep the conversation moving with the best grounded interpretation it can produce, then ask a targeted follow-up or continue retrieval.
