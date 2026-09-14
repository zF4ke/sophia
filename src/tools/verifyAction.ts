import { z } from "zod";
import { ActionVerifier } from "@/runtime/tasks/ActionVerifier";
import { T } from "@/shared/discordTools";
import type { ToolDefinition } from "./types";

export const verifyActionTool: ToolDefinition = {
    name: T.verify_action, catalog: { effect: "read", description: "Check a paused task's unknown Discord action against exact current postconditions.", evidenceRole: "discovery_only" },
    schema: { description: "Verify an unknown action receipt in an owned paused task in this location. Supports exact channel/role/message deletions and bot message content when an exact ID was preserved. Definite observations settle the receipt; partial results, permission errors and unsupported actions remain unresolved. Never executes, retries or resumes the action.", parameters: { type: "object", properties: { task_id: { type: "string" }, action_id: { type: "string" } }, required: ["task_id", "action_id"] } },
    capability: { description: "Read current action postconditions.", sideEffectLevel: "none", inputSchema: z.object({ task_id: z.string(), action_id: z.string() }), outputSchema: z.any(), authRequirements: [], costClass: "normal", latencyClass: "medium", preconditions: [], postconditions: [],
        async run(context, args) { const data = await ActionVerifier.verify(context, String(args.task_id), String(args.action_id)); return { tool: T.verify_action, summary: `Action ${data.actionId}: ${data.resolution}. ${data.detail}`, data }; } },
    strategy: { extractEvidence: () => [] }, display: { icon: "🔎", labelPt: "Verificar ação" },
};
