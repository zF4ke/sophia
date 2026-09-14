import { afterEach, expect, it, vi } from "vitest";
import { buildCostsPanel, handleCostsInteraction } from "@/discord/commands/system/costsPanel";
import { taskStore } from "@/runtime/tasks/TaskStore";
import { SecurityService } from "@/security/SecurityService";
afterEach(() => vi.restoreAllMocks());
it("rejects foreign controls and revoked operators without querying usage", async () => {
    vi.spyOn(SecurityService, "initialize").mockResolvedValue();
    vi.spyOn(SecurityService, "isAdmin").mockReturnValue(false);
    const costs = vi.spyOn(taskStore, "costs");
    const interaction = { customId: "costs:alice:month", user: { id: "bob" }, reply: vi.fn() };
    expect(await handleCostsInteraction(interaction as any)).toBe(true);
    interaction.customId = "settings:costs:bob:month";
    expect(await handleCostsInteraction(interaction as any)).toBe(true);
    await expect(buildCostsPanel("bob", "all")).rejects.toThrow("operadores");
    expect(costs).not.toHaveBeenCalled();
});
it("queries only the viewer and renders unique, bounded controls with unknown costs visible", async () => {
    vi.spyOn(SecurityService, "initialize").mockResolvedValue();
    const totals = { attempts: 2, failures: 1, knownCostUsd: 0, unpriced: 2, inputTokens: 0, outputTokens: 0, missingTokens: 2, background: 0 };
    const costs = vi.spyOn(taskStore, "costs").mockResolvedValue({ window: "month", since: 0, until: Date.now(), totals, models: Array.from({ length: 8 }, (_, i) => ({ ...totals, model: `${i}${"x".repeat(300)}` })) });
    const panel = await buildCostsPanel("alice", "own");
    expect(costs).toHaveBeenCalledWith("alice", "month");
    const json = JSON.stringify(panel.components.map(c => c.toJSON()));
    expect(json).toContain("Custo indisponível");
    const ids = [...json.matchAll(/"custom_id":"([^"]+)"/g)].map(m => m[1]);
    expect(new Set(ids).size).toBe(ids.length);
    expect((panel.components[0].toJSON() as any).components[0].content.length).toBeLessThan(4000);
});
