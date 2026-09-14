---
title: When something fails
sidebar:
  order: 4
---

| Symptom | Check |
| --- | --- |
| Sophia ignores a request | Confirm the location is enabled and the authenticated account has an unexpired grant. Check bot channel visibility and Message Content intent. |
| Reads work but changes fail | Check grant tier, Ask approval ownership, explicit denies, protected resources and Discord role hierarchy. |
| A history answer is incomplete | Ask for coverage and original sources. Refresh or deep-index the relevant channel. Search is not complete history. |
| Workspace unavailable | Run `docker version`, confirm a Linux server is present, and run `npm run sandbox:build`. |
| Docker Desktop is open but the engine is missing | Wait for startup, check WSL 2 and virtualization, and use Docker's diagnostics. Avoid deleting Docker data as a routine repair. |
| Zen returns MissingSessionID | Use the current Sophia Responses adapter. It sends session metadata with Sophia's client identity. |
| Zen returns 401 | Check `OPENCODE_API_KEY` locally and restart after changing it. Do not post the key in chat or logs. |
| An image or audio clip is unsupported | Check profile modalities and adapter support; an expired attachment may need uploading again. |
| Task paused after a restart | Inspect `/tasks` and explicitly resume. Unknown writes must be verified first. |
| Costs are missing | The provider may not have returned usage, prices may be absent, or the associated task may have been forgotten. |
| Docs search is absent in development | Build the site and run `npm run docs:preview` to test the production search index. |

Use `/debug` and task details for diagnosis. Traces can contain conversation content, so inspect and redact them before sharing. Keep the exact error and task ID; avoid including credentials.
