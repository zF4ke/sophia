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
import type { ToolDefinition, CapabilityContext } from "./types";

// ── Import all per-tool definitions ─────────────────────────────────

import { retrieveMessagesTool } from "./retrieveMessages";
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
import { manageMemberRolesTool } from "./manageMemberRoles";
import { sendMessageTool } from "./sendMessage";
import { clearMessagesTool } from "./clearMessages";
import { deleteChannelTool } from "./deleteChannel";

// ── Aggregated tool list ────────────────────────────────────────────

export const ALL_TOOLS: readonly ToolDefinition[] = [
    // ── Retrieval ──
    retrieveMessagesTool,
    // ── Server & channel discovery ──
    listGuildStructureTool,
    getGuildContextTool,
    resolveChannelTargetsTool,
    // ── Member discovery ──
    resolveMemberIdentityTool,
    getMemberProfileTool,
    listMembersTool,
    getRoleInfoTool,
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
    manageMemberRolesTool,
    sendMessageTool,
    // ── Destructive ──
    clearMessagesTool,
    deleteChannelTool,
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

export function isWriteTool(name: string): boolean {
    return _writeSet.has(name);
}

export function isDestructiveTool(name: string): boolean {
    return _destructiveSet.has(name);
}

export function isMutatingTool(name: string): boolean {
    return _writeSet.has(name) || _destructiveSet.has(name);
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
