export const ACCESS_POLICY_TARGETS = {
    mention: "trigger_mention",
    reply: "trigger_reply",
} as const;

export type AccessPolicyTarget =
    (typeof ACCESS_POLICY_TARGETS)[keyof typeof ACCESS_POLICY_TARGETS];

export const ACCESS_POLICY_TARGET_DESCRIPTIONS: Record<AccessPolicyTarget, string> = {
    [ACCESS_POLICY_TARGETS.mention]: "Responder quando alguém menciona a Sophia em mensagens.",
    [ACCESS_POLICY_TARGETS.reply]: "Responder quando alguém responde a uma mensagem da Sophia.",
};

export function isAccessPolicyTarget(value: string): value is AccessPolicyTarget {
    return Object.values(ACCESS_POLICY_TARGETS).includes(value as AccessPolicyTarget);
}
