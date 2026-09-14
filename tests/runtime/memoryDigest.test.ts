import { afterEach, describe, expect, it, vi } from "vitest";
import { buildMemoryDigest } from "@/runtime/memoryDigest";
import { knowledgeStore } from "@/memory/KnowledgeStore";
afterEach(() => vi.restoreAllMocks());
describe("memory discovery hint", () => {
    it("passes the complete audience and includes labels without embedding private fact values", async () => {
        vi.spyOn(knowledgeStore, "identity").mockResolvedValue({ id: "one-sophia", name: "Sophia" });
        const search = vi.spyOn(knowledgeStore, "search").mockResolvedValue([{ key: "Deployment", value: "private-fact" }] as never);
        const hint = await buildMemoryDigest("g1", "u1", "c1");
        expect(search).toHaveBeenCalledWith({ guildId: "g1", actorId: "u1", channelId: "c1" }, "", 5);
        expect(hint).toContain("Deployment");
        expect(hint).toContain("one-sophia");
        expect(hint).not.toContain("private-fact");
    });
    it("supports authenticated DMs and reports unavailable storage honestly", async () => {
        vi.spyOn(knowledgeStore, "identity").mockResolvedValue({ id: "one-sophia", name: "Sophia" });
        vi.spyOn(knowledgeStore, "search").mockResolvedValue([]);
        expect(await buildMemoryDigest(null, "u1", "dm")).toContain("memory_remember");
        vi.spyOn(knowledgeStore, "search").mockRejectedValue(new Error("Disk unavailable"));
        expect(await buildMemoryDigest(null, "u1", "dm")).toBe("Memory status: unavailable.");
    });
});
