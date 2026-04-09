import { Message, MessageFlags } from "discord.js";
import { renderDebugTrace } from "@/discord/debug/renderDebugTrace";
import type { DebugSessionReporter, DebugTraceState } from "@/discord/debug/types";
import type {
    GroundedAnswerMode,
    GroundingDecisionMode,
    GroundingSummary,
    RetrievalControllerDecision,
} from "@/shared/appTypes";

const MAX_EVENTS = 6;
const MAX_PREVIEW_LENGTH = 140;

function normalizePreview(question: string): string {
    const compact = question.replace(/\s+/g, " ").trim();
    if (!compact) {
        return "Sem texto.";
    }

    return compact.length > MAX_PREVIEW_LENGTH
        ? `${compact.slice(0, MAX_PREVIEW_LENGTH - 3)}...`
        : compact;
}

function normalizeMode(mode: "direct_answer" | "discord_grounded"): string {
    return mode === "direct_answer" ? "Resposta direta" : "Com grounding do Discord";
}

function normalizeError(error: unknown): string {
    if (error instanceof Error && error.message) {
        return error.message;
    }

    return "Erro desconhecido";
}

export class DebugSession implements DebugSessionReporter {
    private readonly state: DebugTraceState;
    private updateQueue: Promise<void> = Promise.resolve();

    public constructor(private readonly message: Message, question: string) {
        this.state = {
            questionPreview: normalizePreview(question),
            status: "running",
            stage: "Iniciando",
            mode: null,
            controllerDecision: null,
            toolNames: [],
            groundingSummary: null,
            groundingDecisionMode: null,
            groundedAnswerMode: null,
            contextCacheStatus: "none",
            recentEvents: ["Iniciado"],
            startedAt: Date.now(),
        };
    }

    public async setClassifying(): Promise<void> {
        await this.mutate("Classificando pedido", "Classificando o pedido");
    }

    public async setClassification(
        mode: "direct_answer" | "discord_grounded"
    ): Promise<void> {
        const normalizedMode = normalizeMode(mode);
        await this.mutate(
            "Classificação concluída",
            `Modo escolhido: ${normalizedMode}`,
            (state) => {
                state.mode = normalizedMode;
            }
        );
    }

    public async setPlanning(step: number): Promise<void> {
        await this.mutate(
            "Planejando próxima ação",
            `Planejando passo ${step}`
        );
    }

    public async setRouting(decision: RetrievalControllerDecision): Promise<void> {
        const source = decision.source === "ai" ? "IA" : "determinística";
        const target = decision.targetText ? ` · alvo ${decision.targetText}` : "";
        await this.mutate(
            "Roteando pedido",
            `Controle: ${source} · ${decision.questionIntent}${target}`,
            (state) => {
                state.controllerDecision = decision;
            }
        );
    }

    public async setContextCacheStatus(
        status: "none" | "seeded" | "reused"
    ): Promise<void> {
        const labels = {
            none: "Cache de contexto: não",
            seeded: "Cache de contexto: semeado",
            reused: "Cache de contexto: reutilizado",
        };

        await this.mutate(
            "Verificando cache de contexto",
            labels[status],
            (state) => {
                state.contextCacheStatus = status;
            }
        );
    }

    public async setToolRunning(toolName: string, details: string[] = []): Promise<void> {
        await this.mutate(
            `Usando ${toolName}`,
            `Executando ${toolName}`,
            undefined,
            details
        );
    }

    public async setToolResult(
        toolName: string,
        summary: string,
        itemCount?: number
    ): Promise<void> {
        await this.mutate(
            `Resultado de ${toolName}`,
            itemCount === undefined
                ? `${toolName}: ${summary}`
                : `${toolName}: ${summary} (${itemCount})`,
            (state) => {
                if (!state.toolNames.includes(toolName)) {
                    state.toolNames.push(toolName);
                }
            }
        );
    }

    public async setToolProgress(toolName: string, summary: string): Promise<void> {
        await this.mutate(
            `Usando ${toolName}`,
            `${toolName}: ${summary}`
        );
    }

    public async setGroundingSummary(
        summary: GroundingSummary,
        decisionMode?: GroundingDecisionMode,
        answerMode?: GroundedAnswerMode
    ): Promise<void> {
        const cacheAwareEvent = summary.sufficient
            ? `Base suficiente: mensagens ${summary.messageEvidenceCount} · contexto ao vivo ${summary.liveEvidenceCount}`
            : `Base insuficiente: mensagens ${summary.messageEvidenceCount} · contexto ao vivo ${summary.liveEvidenceCount}`;
        await this.mutate(
            "Avaliando evidências",
            cacheAwareEvent,
            (state) => {
                state.groundingSummary = summary;
                state.groundingDecisionMode = decisionMode || null;
                state.groundedAnswerMode = answerMode || null;
            }
        );
    }

    public async setGenerating(): Promise<void> {
        await this.mutate("Gerando resposta", "Gerando resposta final");
    }

    public async finishSuccess(summary = "Resposta concluída"): Promise<void> {
        await this.mutate(
            "Concluído",
            summary,
            (state) => {
                state.status = "completed";
                state.stage = "Concluído";
            }
        );
    }

    public async finishError(error: unknown): Promise<void> {
        await this.mutate(
            "Falhou",
            `Erro: ${normalizeError(error)}`,
            (state) => {
                state.status = "failed";
                state.stage = "Falhou";
            }
        );
    }

    private async mutate(
        stage: string,
        event: string,
        mutateState?: (state: DebugTraceState) => void,
        extraEvents: string[] = []
    ): Promise<void> {
        this.updateQueue = this.updateQueue
            .then(async () => {
                this.state.stage = stage;
                const normalizedExtraEvents = extraEvents
                    .map((item) => item.trim())
                    .filter(Boolean);
                this.state.recentEvents = [
                    event,
                    ...normalizedExtraEvents,
                    ...this.state.recentEvents,
                ].slice(0, MAX_EVENTS);

                mutateState?.(this.state);

                await this.message.edit({
                    components: [renderDebugTrace(this.state)],
                    flags: MessageFlags.IsComponentsV2,
                });
            })
            .catch((error) => {
                console.error("Error updating debug session:", error);
            });

        await this.updateQueue;
    }
}
