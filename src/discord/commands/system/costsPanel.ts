import { ActionRowBuilder, ButtonBuilder, ButtonStyle, ContainerBuilder, MessageFlags, TextDisplayBuilder, type ButtonInteraction } from "discord.js";
import { taskStore } from "@/runtime/tasks/TaskStore";
import type { CostWindow } from "@/ai/CostReport";
import { SecurityService } from "@/security/SecurityService";
const windows: Record<CostWindow, string> = { day: "24 horas", week: "7 dias", month: "30 dias", all: "Todo o registo" };
const number = (value: number) => value.toLocaleString("pt-PT");
const money = (value: number) => `$${value.toFixed(value > 0 && value < 0.01 ? 6 : 4)} USD`;

export async function buildCostsPanel(actorId: string, scope: "own" | "all", window: CostWindow = "month") {
    await SecurityService.initialize();
    if (scope === "all" && !SecurityService.isAdmin(actorId)) throw new Error("Apenas operadores podem consultar os custos da instalação.");
    const report = await taskStore.costs(scope === "all" ? null : actorId, window);
    const total = report.totals;
    const container = new ContainerBuilder().setAccentColor(0x448879).addTextDisplayComponents(new TextDisplayBuilder().setContent([
        `## Custos · ${windows[window]}`, scope === "all" ? "Toda a instalação" : "A tua utilização em todos os locais",
        `### ${total.attempts === 0 ? "Sem utilização registada" : total.unpriced === total.attempts ? "Custo indisponível" : money(total.knownCostUsd)}`,
        "Estimativa dos pedidos com preço conhecido. Não é uma fatura nem um limite de execução.",
        `${number(total.attempts)} tentativas · ${number(total.failures)} falhas · ${number(total.background)} em segundo plano`,
        `Tokens informados: ${number(total.inputTokens)} de entrada · ${number(total.outputTokens)} de saída`,
        `${number(total.unpriced)} tentativas sem custo calculável · ${number(total.missingTokens)} sem contagem completa de tokens`,
        "### Modelos", ...report.models.map(model => `**${model.model.replace(/[`*_~<>|\n\r]/g, "").slice(0, 100)}**\n${model.unpriced === model.attempts ? "Custo indisponível" : money(model.knownCostUsd)} · ${number(model.attempts)} tentativas · ${number(model.unpriced)} sem preço`),
        "Até 8 modelos, ordenados pelo custo conhecido. Inclui tentativas repetidas e falhadas; falhas sem dados de utilização não são tratadas como gratuitas. Os registos apagados com um pedido deixam de entrar no total.",
        `Atualizado <t:${Math.floor(report.until / 1000)}:R> · períodos móveis em UTC`,
    ].join("\n\n")));
    const prefix = scope === "all" ? "settings:costs" : "costs";
    const buttons = new ActionRowBuilder<ButtonBuilder>().addComponents(...Object.entries(windows).map(([key, label]) =>
        new ButtonBuilder().setCustomId(`${prefix}:${actorId}:${key}`).setLabel(label).setStyle(key === window ? ButtonStyle.Primary : ButtonStyle.Secondary)),
        new ButtonBuilder().setCustomId(`${prefix}:${actorId}:${window}:refresh`).setLabel("Atualizar").setStyle(ButtonStyle.Secondary));
    const back = new ActionRowBuilder<ButtonBuilder>().addComponents(new ButtonBuilder().setCustomId("settings:tab:model").setLabel("Voltar às definições").setStyle(ButtonStyle.Secondary));
    return { components: [container, buttons, ...scope === "all" ? [back] : []], flags: MessageFlags.IsComponentsV2 as const, allowedMentions: { parse: [] as never[] } };
}

export async function handleCostsInteraction(interaction: ButtonInteraction): Promise<boolean> {
    const match = /^(settings:)?costs:([^:]+):(day|week|month|all)(?::refresh)?$/.exec(interaction.customId);
    if (!match) return false;
    if (match[2] !== interaction.user.id) { await interaction.reply({ content: "Este painel pertence a outro utilizador. Abre /costs para consultar o teu.", flags: MessageFlags.Ephemeral }); return true; }
    await SecurityService.initialize();
    if (match[1] && !SecurityService.isAdmin(interaction.user.id)) { await interaction.reply({ content: "O acesso de operador foi revogado.", flags: MessageFlags.Ephemeral }); return true; }
    await interaction.deferUpdate();
    await interaction.editReply(await buildCostsPanel(interaction.user.id, match[1] ? "all" : "own", match[3] as CostWindow));
    return true;
}
