/**
 * Shared query tokenization for FTS5 MATCH searches (libSQL/SQLite).
 *
 * - strips diacritics so "São" matches "sao" (pairs with `remove_diacritics 2`)
 * - keeps only [a-z0-9] per term so the joined query is safe as bare FTS5 syntax
 * - drops terms of length <= 1 to avoid noisy prefix matches
 */
export function tokenizeQueryTerms(query: string): string[] {
    return query
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .split(/\s+/)
        .map((term) => term.replace(/[^a-z0-9]/g, ""))
        .filter((term) => term.length > 1);
}

/** Build a safe bare-token FTS5 MATCH query with prefix matching per term. */
export function buildPrefixFtsQuery(terms: string[]): string {
    return terms.map((term) => `${term}*`).join(" OR ");
}

/** Normalize text the same way queries are tokenized, for lexical scoring. */
export function normalizeForLexicalMatch(text: string): string {
    return text
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase();
}
