import type { z } from "zod";
import type { Guild } from "discord.js";
import type {
    ActiveRetrievalSession,
    EvidenceItem,
    RetrievalSummary,
    SideEffectLevel,
    ToolArguments,
} from "@/runtime/contracts";
import type {
    DiscordToolResult,
    ResolvedMemberIdentity,
    ResolvedChannelTarget,
} from "@/shared/appTypes";
import type { DiscordToolEvidenceRole } from "@/shared/discordTools";

// ── Shared context passed to every capability run() ─────────────────

export type CapabilityContext = {
    guild: Guild | null;
    question: string;
    currentChannelId?: string | null;
    onProgress?: (toolName: string, summary: string) => Promise<void> | void;
};

// ── Tool effect classification ──────────────────────────────────────

export type ToolEffect = "read" | "write" | "destructive";

// ── Per-tool definition ─────────────────────────────────────────────

export interface ToolDefinition {
    /** Unique tool name (should match a DiscordToolName literal). */
    name: string;

    /** Catalog metadata: classification, description, evidence role. */
    catalog: {
        effect: ToolEffect;
        description: string;
        evidenceRole: DiscordToolEvidenceRole;
    };

    /** JSON Schema for the model (native function calling). */
    schema: {
        description: string;
        parameters: Record<string, unknown>;
    };

    /** Runtime capability implementation. */
    capability: {
        description: string;
        inputSchema: z.ZodTypeAny;
        outputSchema: z.ZodTypeAny;
        sideEffectLevel: SideEffectLevel;
        authRequirements: string[];
        costClass: "cheap" | "normal" | "expensive";
        latencyClass: "fast" | "medium" | "slow";
        preconditions: string[];
        postconditions: string[];
        run(context: CapabilityContext, args: ToolArguments): Promise<DiscordToolResult>;
    };

    /** Evidence extraction strategy. */
    strategy: {
        extractEvidence(run: DiscordToolResult): EvidenceItem[];
        extractResolvedMember?(run: DiscordToolResult): ResolvedMemberIdentity | null;
        extractResolvedChannel?(run: DiscordToolResult): ResolvedChannelTarget | null;
        extractRetrievalSummary?(run: DiscordToolResult): RetrievalSummary | null;
        extractRetrievalSession?(run: DiscordToolResult): ActiveRetrievalSession | null;
    };

    /** UI display info (icon + Portuguese label). */
    display: { icon: string; labelPt: string };

    /** Approval description generator (only for write/destructive tools). */
    describeApproval?: (args: ToolArguments) => string;
}
