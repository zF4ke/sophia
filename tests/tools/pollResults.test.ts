import { beforeEach, describe, expect, it } from "vitest";
import { CapabilityRegistry } from "@/capabilities/CapabilityRegistry";

function makePollMessage(id: string, votes: number[], expiresTimestamp: number | null = 1_800_000_000_000) {
    return {
        id,
        poll: {
            question: { text: "Julgamento de Sophia: inocente ou culpada?" },
            allowMultiselect: false,
            expiresTimestamp,
            answers: new Map(
                votes.map((voteCount, index) => [index, {
                    id: index,
                    text: `Opção ${index + 1}`,
                    voteCount,
                    emoji: null,
                }]),
            ),
            fetch: async function (this: unknown) { return this; },
        },
    };
}

function makeContext(messages: unknown[]) {
    const channel = {
        id: "c1",
        messages: {
            fetch: (query: unknown) => {
                if (typeof query === "string") {
                    const found = messages.find((m) => (m as { id: string }).id === query);
                    return found ? Promise.resolve(found) : Promise.reject(new Error("Unknown Message"));
                }
                return Promise.resolve(messages);
            },
        },
    };
    return {
        guild: { id: "g1", channels: { cache: new Map([["c1", channel]]) } } as never,
        question: "poll votes?",
        currentChannelId: "c1",
    };
}

describe("get_poll_results tool", () => {
    beforeEach(() => {
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
    });

    it("returns per-answer counts, percentages, and the leader", async () => {
        const context = makeContext([makePollMessage("m1", [1, 1])]);
        const result = await CapabilityRegistry.get("get_poll_results").run(context, {} as never);

        expect(result.errorMessage).toBeUndefined();
        expect((result.data as any).totalVotes).toBe(2);
        expect((result.data as any).answers).toEqual([
            { id: 0, text: "Opção 1", emoji: null, votes: 1, percentage: 50 },
            { id: 1, text: "Opção 2", emoji: null, votes: 1, percentage: 50 },
        ]);
        expect(result.summary).toContain("2 vote(s)");
    });

    it("finds the newest poll in the channel when no message_id is given", async () => {
        // Discord returns newest-first; the newest poll wins over older ones.
        const context = makeContext([
            { id: "m-text", poll: null },
            makePollMessage("m-new", [0, 0, 2]),
            makePollMessage("m-old", [5]),
        ]);
        const result = await CapabilityRegistry.get("get_poll_results").run(context, {} as never);

        expect(result.errorMessage).toBeUndefined();
        expect((result.data as any).messageId).toBe("m-new");
        expect((result.data as any).answers[2].votes).toBe(2);
    });

    it("reports zero votes cleanly and tolerates a failing refresh", async () => {
        const message = makePollMessage("m1", [0, 0], null);
        (message.poll as { fetch: () => Promise<unknown> }).fetch = async () => {
            throw new Error("refresh unavailable");
        };
        const context = makeContext([message]);
        const result = await CapabilityRegistry.get("get_poll_results").run(context, { message_id: "m1" } as never);

        expect(result.errorMessage).toBeUndefined();
        expect((result.data as any).totalVotes).toBe(0);
        expect(result.summary).toContain("0 vote(s)");
    });

    it("rejects messages without a poll", async () => {
        const context = makeContext([{ id: "m-text", poll: null }]);
        const result = await CapabilityRegistry.get("get_poll_results").run(context, { message_id: "m-text" } as never);

        expect(result.errorMessage).toMatch(/no poll/i);
    });
});
