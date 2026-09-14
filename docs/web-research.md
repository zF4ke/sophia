# Web research

`web_search` uses Brave when its secret API key is configured, otherwise DuckDuckGo HTML search. Each snippet comes from the same result block as its URL. An unavailable or unrecognized provider response is an error, not a claim that no sources exist. The former Google HTML scraper and the misleading "always free" output field are removed.

`fetch_url` opens a public page and preserves its extracted text in the owning task. Results include the requested URL, final URL, title, capture timestamp, source ID, total length and exact next offset. `source_read` reads more of that captured source without another network request. This is task evidence; other actors, tasks and locations cannot reuse the handle.

`SafeWebClient` accepts public HTTP(S) destinations on standard ports, rejects URL credentials, validates all DNS answers and pins each connection to a validated address. Redirects receive the same checks. Provider credentials are not forwarded across redirects. Requests have a 20-second deadline and a 2 MiB transport limit. Private, loopback, link-local, reserved and mapped addresses are excluded. The client does not send cookies, execute scripts or load page subresources.

`ReadablePage` uses [LinkeDOM](https://github.com/WebReflection/linkedom) to parse HTML and preserve readable structure and links. HTTP connections use the [Node request API](https://nodejs.org/api/http.html). Unsupported binary content belongs in the isolated file workflow. Pages that require JavaScript or return unsupported compression report that limitation.

The deterministic suite tests destination checks, redirects, credential stripping, snippet attribution and owned pagination. A separate opt-in test verifies a real HTTPS page through the pinned transport. Search-provider availability can still vary and must not be confused with a completed investigation.
