// Token counting for pre-send prompt measurement.
//
// Uses gpt-tokenizer (cl100k_base BPE) as a provider-agnostic approximation.
// Exact counts will differ per provider (Gemini, Anthropic, etc. each use
// their own tokenizer), but for threshold checks — "is this prompt too big?" —
// cl100k is within ~10–15% of real counts across modern models, which is
// plenty for compaction routing decisions.
//
// Wrapped behind a single function so we can swap implementations later
// (e.g. per-provider tokenizers) without touching call sites.

import { encode } from "gpt-tokenizer";

export function countTokens(text: string): number {
    if (!text) return 0;
    try {
        return encode(text).length;
    } catch {
        // Fallback: rough char-based estimate if tokenizer fails for any reason.
        return Math.ceil(text.length / 4);
    }
}
