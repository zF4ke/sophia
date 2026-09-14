import { z } from "zod";
import { DISCORD_TOOL_NAMES } from "@/shared/discordTools";

const grant = {
    level: z.enum(["read", "write", "destructive"]).default("read"),
    mode: z.enum(["ask", "auto"]).default("ask"),
    expiresAt: z.string().datetime({ offset: true }).optional(),
};
export const accessConfigSchema = z.object({
    directMessages: z.boolean().default(false),
    rules: z.array(z.object({
        subject: z.enum(["user", "role"]),
        subjectId: z.string().min(1),
        guildId: z.string().min(1).optional(),
        tool: z.string().refine(value => DISCORD_TOOL_NAMES.includes(value as never), "Unknown capability").optional(),
        tier: z.enum(["read", "write", "destructive"]).optional(),
        decision: z.enum(["allow", "ask", "deny"]),
    }).strict().refine(rule => !!rule.tool || !!rule.tier, "A rule needs a capability or action tier")
        .refine(rule => rule.subject !== "role" || !!rule.guildId, "Role rules need a guild")).optional(),
    users: z.array(z.object({
        userId: z.string().min(1),
        guildId: z.string().min(1).optional(),
        ...grant,
    }).strict()).default([]),
    roles: z.array(z.object({
        guildId: z.string().min(1),
        roleId: z.string().min(1),
        ...grant,
    }).strict()).default([]),
}).strict();
export type AccessConfig = z.infer<typeof accessConfigSchema>;
export type AccessDecision = "allow" | "ask" | "deny";
