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

function formatAnswerMode(state: DebugTraceState): string {
    if (!state.groundedAnswerMode) {
        return "";
    }

    if (state.groundedAnswerMode === "confident") {
        return "confiante";
    }

    if (state.groundedAnswerMode === "best_effort") {
        return "melhor esforço";
    }

    return "insuficiente";
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

function formatControllerDecision(state: DebugTraceState): string {
    if (!state.controllerDecision) {
        return "";
    }

    const source = state.controllerDecision.source === "ai" ? "IA" : "determinística";
    const target = state.controllerDecision.targetText
        ? ` · alvo ${state.controllerDecision.targetText}`
        : "";
    return `${source} · ${state.controllerDecision.questionIntent}${target}`;
}

export function renderDebugTrace(state: DebugTraceState): ContainerBuilder {
    const tools = state.toolNames.length ? state.toolNames.join(", ") : "nenhuma";
    const recentEvents = state.recentEvents.length
        ? state.recentEvents.map((event) => `- ${event}`).join("\n")
        : "- Aguardando etapas.";
    const details = [
        `**Pergunta:** ${state.questionPreview}`,
        `**Estado:** ${formatStatus(state.status)}`,
        `**Etapa:** ${state.stage}`,
        `**Modo:** ${state.mode ?? "a decidir"}`,
    ];

    if (state.mode === "Com grounding do Discord") {
        const route = formatControllerDecision(state);
        if (route) {
            details.push(`**Controle:** ${route}`);
        }

        details.push(`**Cache de contexto:** ${formatContextCacheStatus(state)}`);
        details.push(`**Ferramentas:** ${tools}`);

        if (state.groundingSummary) {
            details.push(`**Base útil:** ${formatGroundingSummary(state)}`);
            details.push(`**Grounding:** ${formatGroundingState(state)}`);
        }

        const answerMode = formatAnswerMode(state);
        if (answerMode) {
            details.push(`**Resultado:** ${answerMode}`);
        }

        if (state.groundingDecisionMode) {
            details.push(`**Decisão:** ${formatGroundingDecisionMode(state)}`);
        }
    } else {
        details.push(`**Ferramentas:** ${tools}`);
    }

    details.push(`**Tempo:** ${formatDuration(state.startedAt)}`);

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
            new TextDisplayBuilder().setContent(details.join("\n"))
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
