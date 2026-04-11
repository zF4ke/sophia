import { describe, expect, it } from "vitest";
import { planWithModel } from "@/runtime/planning";
import type { TurnInput } from "@/runtime/contracts";

const LIVE_MODEL_TESTS_ENABLED =
    process.env.LIVE_MODEL_TESTS === "1" &&
    !!process.env.OPENROUTER_API_KEY &&
    process.env.OPENROUTER_API_KEY !== "test-key";

const describeLive = LIVE_MODEL_TESTS_ENABLED ? describe : describe.skip;

function createInput(overrides: Partial<TurnInput> = {}): TurnInput {
    return {
        question: "test",
        user: { id: "u-requester" } as any,
        requesterDisplayName: "Requester",
        guild: { id: "g1", name: "Oz Synthesis" } as any,
        currentChannelId: "c1",
        nativeThreadId: null,
        requestedWebMode: "off",
        trigger: "talk",
        replyContext: null,
        referencedMessage: null,
        conversation: {
            key: "g1:c1:channel",
            kind: "channel",
            trigger: "talk",
            replyAnchorMessageId: null,
            nativeThreadId: null,
        },
        ...overrides,
    };
}

describeLive("live planner behavior", () => {
    it(
        "keeps a greeting conversational with the real model",
        async () => {
            process.env.DISCORD_TOKEN ||= "test-token";

            const result = await planWithModel(
                createInput({
                    question: "Olá tudo bem?",
                }),
                [{ authorName: "Requester", content: "@Sophia Olá tudo bem?", createdTimestamp: Date.now() }]
            );

            expect(result.mode).toBe("conversation");
            expect(result.candidateCapabilities).toEqual([]);
        },
        120000
    );

    it(
        "chooses a current-guild research plan for exact member and channel references",
        async () => {
            process.env.DISCORD_TOKEN ||= "test-token";

            const result = await planWithModel(
                createInput({
                    question:
                        "O que <@111111111111111111> disse em <#123456789012345678> sobre a imagem?",
                    trigger: "mention",
                })
            );

            expect(result.mode).toBe("research");
            expect(result.candidateCapabilities).toContain("retrieve_messages");
            expect(
                result.candidateCapabilities.some(
                    (capability) =>
                        capability === "resolve_member_identity" ||
                        capability === "resolve_channel_targets"
                )
            ).toBe(true);
        },
        120000
    );
});
