import type { DiscordToolName } from "@/shared/discordTools";
import type {
    CapabilityManifest,
    EvidenceItem,
    ActiveRetrievalSession,
    RetrievalSummary,
    ToolArguments,
} from "@/runtime/contracts";
import type {
    DiscordToolResult,
    ResolvedMemberIdentity,
    ResolvedChannelTarget,
} from "@/shared/appTypes";
import type { ToolDefinition, CapabilityContext, ToolEffect } from "./types";

// ── Import all per-tool definitions ─────────────────────────────────

import { retrieveMessagesTool } from "./retrieveMessages";
import { searchMessagesTool } from "./searchMessages";
import { randomChannelMessageTool } from "./randomChannelMessage";
import { listGuildStructureTool } from "./listGuildStructure";
import { getGuildContextTool } from "./getGuildContext";
import { resolveChannelTargetsTool } from "./resolveChannelTargets";
import { resolveMemberIdentityTool } from "./resolveMemberIdentity";
import { getMemberProfileTool } from "./getMemberProfile";
import { listMembersTool } from "./listMembers";
import { getRoleInfoTool } from "./getRoleInfo";
import { listThreadsTool } from "./listThreads";
import { readThreadMessagesTool } from "./readThreadMessages";
import { measureTextLengthTool } from "./measureTextLength";
import { evaluateMathTool } from "./evaluateMath";
import { createChannelTool } from "./createChannel";
import { createCategoryTool } from "./createCategory";
import { createThreadTool } from "./createThread";
import { moveChannelTool } from "./moveChannel";
import { moveCategoryTool } from "./moveCategory";
import { manageMemberRolesTool } from "./manageMemberRoles";
import { sendMessageTool } from "./sendMessage";
import { editMessageTool } from "./editMessage";
import { createRoleTool } from "./createRole";
import { editChannelTool } from "./editChannel";
import { editRoleTool } from "./editRole";
import { listRolesTool } from "./listRoles";
import { clearMessagesTool } from "./clearMessages";
import { deleteMessagesTool } from "./deleteMessages";
import { deleteChannelTool } from "./deleteChannel";
import { deleteRoleTool } from "./deleteRole";
import { startLongTaskTool } from "./startLongTask";
import { noteAddTool, noteListTool, noteClearTool, planUpdateTool } from "./notes";
import { webSearchTool, fetchUrlTool } from "./webSearch";
import { memoryRememberTool, memorySearchTool } from "./longTermMemory";
import { workflowCreateTool, workflowListTool, workflowRunTool } from "./workflows";
import { toolSearchTool } from "./toolSearch";
import { createPollTool } from "./createPoll";
import { getToolExposure, isDirectTool } from "./toolExposure";

// ── Aggregated tool list ────────────────────────────────────────────

export const ALL_TOOLS: readonly ToolDefinition[] = [
    // ── Retrieval ──
    retrieveMessagesTool,
    searchMessagesTool,
    randomChannelMessageTool,
    // ── Server & channel discovery ──
    listGuildStructureTool,
    getGuildContextTool,
    resolveChannelTargetsTool,
    // ── Member discovery ──
    resolveMemberIdentityTool,
    getMemberProfileTool,
    listMembersTool,
    getRoleInfoTool,
    // ── Role discovery ──
    listRolesTool,
    // ── Thread reading ──
    listThreadsTool,
    readThreadMessagesTool,
    // ── Utilities ──
    measureTextLengthTool,
    evaluateMathTool,
    // ── Write ──
    createChannelTool,
    createCategoryTool,
    createThreadTool,
    moveChannelTool,
    moveCategoryTool,
    manageMemberRolesTool,
    sendMessageTool,
    editMessageTool,
    createRoleTool,
    // ── Destructive ──
    clearMessagesTool,
    deleteMessagesTool,
    deleteChannelTool,
    deleteRoleTool,
    editChannelTool,
    editRoleTool,
    // ── Control ──
    startLongTaskTool,
    // ── Scratchpad ──
    noteAddTool,
    noteListTool,
    noteClearTool,
    planUpdateTool,
    // ── Web ──
    webSearchTool,
    fetchUrlTool,
    // ── Long-term memory ──
    memoryRememberTool,
    memorySearchTool,
    // ── Workflows ──
    workflowCreateTool,
    workflowListTool,
    workflowRunTool,
    // ── Meta (discovery) ──
    toolSearchTool,
    // ── Poll ──
    createPollTool,
];

const TOOL_MAP = new Map<string, ToolDefinition>(
    ALL_TOOLS.map((t) => [t.name, t]),
);

// ── Effect queries ──────────────────────────────────────────────────

const _writeSet = new Set<string>();
const _destructiveSet = new Set<string>();
for (const tool of ALL_TOOLS) {
    if (tool.catalog.effect === "write") _writeSet.add(tool.name);
    if (tool.catalog.effect === "destructive")
        _destructiveSet.add(tool.name);
}

export function isDestructiveTool(name: string): boolean {
    return _destructiveSet.has(name);
}

export function isMutatingTool(name: string): boolean {
    return _writeSet.has(name) || _destructiveSet.has(name);
}

export function getToolEffect(name: string): ToolEffect | undefined {
    return TOOL_MAP.get(name)?.catalog.effect;
}

// ── Display helpers ─────────────────────────────────────────────────

export function getToolDisplay(
    toolName: string,
): { icon: string; labelPt: string } {
    return (
        TOOL_MAP.get(toolName)?.display ?? {
            icon: "🛠️",
            labelPt: toolName,
        }
    );
}

export function describeApproval(
    toolName: string,
    args: ToolArguments,
): string {
    const tool = TOOL_MAP.get(toolName);
    if (tool?.describeApproval) {
        return tool.describeApproval(args);
    }
    return `Executar ${toolName}`;
}

// ── Strategy access ─────────────────────────────────────────────────

export interface ToolStrategy {
    readonly id: DiscordToolName;
    extractEvidence(run: DiscordToolResult): EvidenceItem[];
    extractResolvedMember?(
        run: DiscordToolResult,
    ): ResolvedMemberIdentity | null;
    extractResolvedChannel?(
        run: DiscordToolResult,
    ): ResolvedChannelTarget | null;
    extractRetrievalSummary?(
        run: DiscordToolResult,
    ): RetrievalSummary | null;
    extractRetrievalSession?(
        run: DiscordToolResult,
    ): ActiveRetrievalSession | null;
}

export function getToolStrategy(id: DiscordToolName): ToolStrategy {
    const tool = TOOL_MAP.get(id);
    if (!tool) {
        throw new Error(`No tool strategy registered for "${id}"`);
    }
    return {
        id: id,
        ...tool.strategy,
    };
}

// ── Capability access ───────────────────────────────────────────────

export type RuntimeCapability = CapabilityManifest & {
    run(
        context: CapabilityContext,
        args: ToolArguments,
    ): Promise<DiscordToolResult>;
};

function toRuntimeCapability(tool: ToolDefinition): RuntimeCapability {
    return {
        id: tool.name as DiscordToolName,
        kind: "tool",
        description: tool.capability.description,
        inputSchema: tool.capability.inputSchema,
        outputSchema: tool.capability.outputSchema,
        sideEffectLevel: tool.capability.sideEffectLevel,
        authRequirements: tool.capability.authRequirements,
        costClass: tool.capability.costClass,
        latencyClass: tool.capability.latencyClass,
        evidenceRole: tool.catalog.evidenceRole,
        preconditions: tool.capability.preconditions,
        postconditions: tool.capability.postconditions,
        run: tool.capability.run,
    };
}

export function getCapability(id: DiscordToolName): RuntimeCapability {
    const tool = TOOL_MAP.get(id);
    if (!tool) {
        throw new Error(`Unknown capability "${id}".`);
    }
    return toRuntimeCapability(tool);
}

export function listCapabilityManifests(): CapabilityManifest[] {
    return ALL_TOOLS.map((tool) => {
        const { run, ...manifest } = toRuntimeCapability(tool);
        return manifest;
    });
}

export function describeCapabilitiesForPrompt(): string {
    return listCapabilityManifests()
        .map(
            (cap) =>
                `- **${cap.id}** (${cap.sideEffectLevel}): ${cap.description}`,
        )
        .join("\n");
}

// ── Tool schema builder ─────────────────────────────────────────────

export interface NativeToolDef {
    type: "function";
    function: {
        name: string;
        description: string;
        parameters: Record<string, unknown>;
    };
}

export function buildToolDefinitions(): NativeToolDef[] {
    return ALL_TOOLS.map((tool) => ({
        type: "function" as const,
        function: {
            name: tool.name,
            description: tool.schema.description,
            parameters: tool.schema.parameters,
        },
    }));
}

export function buildVisibleToolDefinitions(discovered: Set<string> = new Set()): NativeToolDef[] {
    return ALL_TOOLS.filter((tool) => {
        const exposure = getToolExposure(tool.name as never);
        if (exposure === "direct") return true;
        return discovered.has(tool.name);
    }).map((tool) => ({
        type: "function" as const,
        function: {
            name: tool.name,
            description: tool.schema.description,
            parameters: tool.schema.parameters,
        },
    }));
}

export function getDeferredTools(): typeof ALL_TOOLS {
    return ALL_TOOLS.filter((t) => !isDirectTool(t.name as never)) as unknown as typeof ALL_TOOLS;
}

export function formatDeferredInventory(): string {
    const deferred = getDeferredTools();
    if (!deferred.length) return "All tools are directly available.";
    const byEffect: Record<string, typeof deferred> = {};
    for (const t of deferred) {
        const k = t.catalog.effect;
        if (!byEffect[k]) byEffect[k] = [] as unknown as typeof deferred;
        (byEffect[k] as unknown as typeof t[]).push(t);
    }
    const lines: string[] = ["Deferred tools (call tool_search to load):"];
    for (const [effect, tools] of Object.entries(byEffect)) {
        lines.push(`- ${effect}: ${tools.map((t) => `${t.name} — ${t.catalog.description.slice(0, 80)}`).join("; ")}`);
    }
    return lines.join("\n");
}
