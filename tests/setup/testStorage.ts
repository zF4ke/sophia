import "./storageRoot";
import http from "node:http";
import https from "node:https";

// Deterministic tests must never fall through to a paid provider or Discord.
globalThis.fetch = async () => { throw new Error("Network disabled in deterministic tests. Mock the dependency or use test:live."); };
http.request = (() => { throw new Error("HTTP disabled in deterministic tests. Mock the transport or use test:live."); }) as typeof http.request;
https.request = (() => { throw new Error("HTTPS disabled in deterministic tests. Mock the transport or use test:live."); }) as typeof https.request;
