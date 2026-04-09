import { describe, expect, it } from "vitest";
import {
    extractLikelyPersonName,
    extractMentionedChannelIds,
    extractRequestedOrdinal,
    isGuildContextQuestion,
    isMemberDiscoveryQuestion,
    isPersonCentricQuestion,
} from "@/agent/orchestration/questionAnalysis";

describe("questionAnalysis", () => {
    it("extracts ordinals from Portuguese text", () => {
        expect(extractRequestedOrdinal("Quem é o décimo sétimo membro?")).toBe(17);
        expect(extractRequestedOrdinal("quem é o 21º membro?")).toBe(21);
    });

    it("detects guild-context questions", () => {
        expect(isGuildContextQuestion("em que servidor estou?")).toBe(true);
        expect(isGuildContextQuestion("qual foi a decisão do roadmap?")).toBe(false);
    });

    it("detects member and person-centric questions", () => {
        expect(isMemberDiscoveryQuestion("quem é o terceiro membro?")).toBe(true);
        expect(isPersonCentricQuestion("perfil do scart")).toBe(true);
        expect(extractLikelyPersonName("perfil do scart")).toBe("scart");
        expect(isPersonCentricQuestion("escolha uma musica legal ai do scart")).toBe(false);
    });

    it("extracts channel mentions from discord text", () => {
        expect(extractMentionedChannelIds("escolha uma musica do <#12345>")).toEqual(["12345"]);
    });
});
