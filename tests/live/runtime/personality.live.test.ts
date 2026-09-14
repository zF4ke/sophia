import { afterEach, expect, it, vi } from "vitest";
import { randomUUID } from "node:crypto";
import { Runtime } from "@/runtime/Runtime";
import { boundLiveModelCalls } from "./modelBudget";

afterEach(() => vi.restoreAllMocks());
const live = process.env.LIVE_MODEL_TESTS === "1" ? it : it.skip;
const cases = [
    { name: "English correction", question: "That sounded like a customer support email. Just talk to me normally.", max: 400 },
    { name: "Portuguese conversation", question: "Sophia, esse plano de fazer tudo na véspera tem tudo para correr bem, né?", max: 500 },
    { name: "useful technical explanation", question: "Explain why a JavaScript closure can retain a variable after its outer function returns. Give a small example and explain the output. No follow-up question.", max: 2400 },
    { name: "exact code preservation", question: 'Show this exact JavaScript line in a code block, then explain it in one plain sentence: const label = "alpha—beta";', max: 650 },
];
for (const scenario of cases) live(`Sophia voice: ${scenario.name}`, async () => {
    const usage = boundLiveModelCalls(3);
    const id = randomUUID();
    const result = await Runtime.answer({ question: scenario.question, user: { id } as never, requesterDisplayName: "Tester", guild: null, currentChannelId: id,
        trigger: "mention", authorize: async effect => effect === "none" ? "allow" : "deny", conversation: { key: id, kind: "channel", trigger: "mention", replyAnchorMessageId: null, nativeThreadId: null } });
    expect(result.outcome).toBe("completed");
    expect(result.answer.length).toBeLessThan(scenario.max);
    const prose = result.answer.replace(/```[\s\S]*?```/g, "").replace(/`[^`]*`/g, "");
    expect(prose).not.toMatch(/[—–]/);
    expect(prose).not.toMatch(/let me know|would you like me to|great question|certainly!|anything else|se precisares de mais|o que achas\?/i);
    if (scenario.name === "exact code preservation") expect(result.answer).toContain('const label = "alpha—beta";');
    if (scenario.name === "useful technical explanation") expect(result.answer).toContain("```");
    if (scenario.name === "English correction") expect(result.answer).not.toMatch(/\?/);
    console.info(JSON.stringify({ scenario: scenario.name, ...usage(), answer: result.answer }));
});
