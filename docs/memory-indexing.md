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
- Live fallback crawl uses the official Discord API to fetch channel messages, ingests them through the same memory service, and then reruns local retrieval.
- Channel crawl state is stored so the agent can distinguish indexed channels from readable channels that have not been crawled yet.

## Operational Commands

- `/index status`
- `/index backfill_channel`
- `/index backfill_category`
- `/index repair`
- `/index clear`

If retrieval ranking changes, update the tests and this document in the same change.
