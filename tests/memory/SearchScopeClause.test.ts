import { describe, expect, it } from "vitest";
import { SearchScopeClause } from "@/memory/search/SearchScopeClause";

describe("SearchScopeClause.toFtsQuery", () => {
    it("sanitizes hyphenated terms into safe FTS tokens", () => {
        expect(SearchScopeClause.toFtsQuery("obra-prima")).toBe('"obra" OR "prima"');
    });

    it("drops punctuation and quotes tokens safely", () => {
        expect(SearchScopeClause.toFtsQuery('quais musicas vc achou no scart?')).toBe(
            '"quais" OR "musicas" OR "vc" OR "achou" OR "no" OR "scart"'
        );
    });
});
