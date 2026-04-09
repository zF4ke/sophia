import {
    ContainerBuilder,
    SeparatorBuilder,
    SeparatorSpacingSize,
    TextDisplayBuilder,
} from "discord.js";
import type { DebugTraceState } from "@/discord/debug/types";

function formatDuration(startedAt: number): string {
    const elapsedMs = Math.max(0, Date.now() - startedAt);
    const elapsedSeconds = Math.max(1, Math.round(elapsedMs / 1000));
    return `${elapsedSeconds}s`;
}

function formatStatus(status: DebugTraceState["status"]): string {
    if (status === "completed") {
        return "Concluído";
    }

    if (status === "failed") {
        return "Falhou";
    }

    return "Em andamento";
}

function formatGroundingSummary(state: DebugTraceState): string {
    if (!state.groundingSummary) {
        return "ainda avaliando";
    }

    return `mensagens ${state.groundingSummary.messageEvidenceCount} · contexto ao vivo ${state.groundingSummary.liveEvidenceCount}`;
}

function formatGroundingState(state: DebugTraceState): string {
    if (!state.groundingSummary) {
        return "ainda avaliando";
    }

    return state.groundingSummary.sufficient ? "suficiente" : "insuficiente";
}

function formatGroundingDecisionMode(state: DebugTraceState): string {
    if (!state.groundingDecisionMode) {
        return "ainda avaliando";
    }

    if (state.groundingDecisionMode === "judge") {
        return "juiz";
    }

    if (state.groundingDecisionMode === "reused") {
        return "contexto reutilizado";
    }

    return "heurística";
}

function formatContextCacheStatus(state: DebugTraceState): string {
    if (state.contextCacheStatus === "reused") {
        return "reutilizado";
    }

    if (state.contextCacheStatus === "seeded") {
        return "semeado";
    }

    return "não";
}

function formatRouteDecision(state: DebugTraceState): string {
    if (!state.routeDecision) {
        return "a decidir";
    }

    const source = state.routeDecision.source === "ai" ? "IA" : "determinística";
    const target = state.routeDecision.targetText
        ? ` · alvo ${state.routeDecision.targetText}`
        : "";
    return `${source} · ${state.routeDecision.intent}${target}`;
}

export function renderDebugTrace(state: DebugTraceState): ContainerBuilder {
    const tools = state.toolNames.length ? state.toolNames.join(", ") : "nenhuma";
    const recentEvents = state.recentEvents.length
        ? state.recentEvents.map((event) => `- ${event}`).join("\n")
        : "- Aguardando etapas.";

    return new ContainerBuilder()
        .setAccentColor(
            state.status === "failed"
                ? 0xed4245
                : state.status === "completed"
                  ? 0x57f287
                  : 0x9aa7ff
        )
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent("## Debug da Sophia"),
            new TextDisplayBuilder().setContent(
                [
                    `**Pergunta:** ${state.questionPreview}`,
                    `**Estado:** ${formatStatus(state.status)}`,
                    `**Etapa:** ${state.stage}`,
                    `**Modo:** ${state.mode ?? "a decidir"}`,
                    `**Rota:** ${formatRouteDecision(state)}`,
                    `**Cache de contexto:** ${formatContextCacheStatus(state)}`,
                    `**Ferramentas:** ${tools}`,
                    `**Base útil:** ${formatGroundingSummary(state)}`,
                    `**Grounding:** ${formatGroundingState(state)}`,
                    `**Decisão:** ${formatGroundingDecisionMode(state)}`,
                    `**Tempo:** ${formatDuration(state.startedAt)}`,
                ].join("\n")
            )
        )
        .addSeparatorComponents(
            new SeparatorBuilder()
                .setDivider(true)
                .setSpacing(SeparatorSpacingSize.Small)
        )
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent("### Passos recentes"),
            new TextDisplayBuilder().setContent(recentEvents)
        );
}
