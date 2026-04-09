# Cleanup And Migration Notes

This codebase intentionally removed older Gemini-era and novelty-era paths that conflicted with the new architecture.

## Removed Or Replaced

- Large inline prompt blobs were replaced by `resources/prompts/`.
- The old Gemini-only AI facade was replaced by `ModelGateway` plus agent orchestration.
- Context stuffing and fake web-search behavior were removed from the active runtime.
- The monolithic security and UI services were split into smaller modules.
- Legacy novelty commands and dead helper modules were removed.

## Current Direction

Sophia is now a grounded Discord assistant first. If evidence is weak, the bot should say so plainly instead of inventing answers.
