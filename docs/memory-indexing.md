# Memory And Indexing

Sophia stores Discord messages locally so retrieval quality does not depend on Discord search.

## Write Path

- `src/discord/events/message/messageCreate.event.ts` ingests eligible new messages.
- `src/memory/ingest/` normalizes messages and filters out junk.
- `src/memory/index/` splits messages into chunks.
- `src/memory/repositories/MemoryWriteRepository.ts` writes messages, chunks, embeddings, and index state.

## Read Path

- `src/memory/search/MemorySearchService.ts` performs lexical plus embedding retrieval.
- `src/memory/repositories/MemoryReadRepository.ts` handles thread reads, summaries, and historical lookup.
- `src/memory/DiscordMemoryService.ts` is the small facade used by the rest of the app.

## Operational Commands

- `/index status`
- `/index backfill_channel`
- `/index backfill_category`
- `/index repair`
- `/index clear`

If retrieval ranking changes, update the tests and this document in the same change.
