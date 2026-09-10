import {
    randomUUID,
} from "crypto";
import {
    AnySelectMenuInteraction,
    ButtonInteraction,
    MessageFlags,
    StringSelectMenuInteraction,
} from "discord.js";
import { buildArtifactComponents, type ArtifactSpec } from "./ArtifactBuilder";
import { ArtifactStore } from "./ArtifactStore";
import { runArtifactScript } from "./ArtifactScript";

// Current format: "artifact:<nonce>:<action>". Legacy cards emitted
// "<12-hex nonce>:<action>" with no prefix; both are handled so every
// card sent before the prefix existed keeps working.
const NAV_PATTERN = /^(?:artifact:)?([0-9a-f]{12}):(prev|next|tab|count)$/;
const CUSTOM_PREFIX_PATTERN = /^(?:action|game):/;

export function isArtifactCustomId(customId: string): boolean {
    return NAV_PATTERN.test(customId) || CUSTOM_PREFIX_PATTERN.test(customId);
}

/**
 * Central handler for artifact card interactions. Unlike per-message
 * collectors, this reads the persisted spec and view state from the store,
 * so pagination and tab buttons keep working after a bot restart, for the
 * whole life of the message.
 *
 * Returns true when the interaction belonged to an artifact card.
 */
export async function handleArtifactInteraction(
    interaction: ButtonInteraction | AnySelectMenuInteraction,
): Promise<boolean> {
    const customId = interaction.customId;
    const navMatch = NAV_PATTERN.exec(customId);
    const isCustomAction = CUSTOM_PREFIX_PATTERN.test(customId);
    if (!navMatch && !isCustomAction) return false;
    const action = navMatch ? navMatch[2] : null;

    const row = await ArtifactStore.get(interaction.message.id).catch(() => null);
    if (!row || !row.specJson) {
        await interaction
            .reply({ content: "Este cartão perdeu o estado guardado. Pede para reenviar ou editar o artefacto.", flags: MessageFlags.Ephemeral })
            .catch(() => undefined);
        return true;
    }
    if (row.guildId && interaction.guildId && row.guildId !== interaction.guildId) {
        await interaction.deferUpdate().catch(() => undefined);
        return true;
    }

    let spec: ArtifactSpec;
    try {
        const parsed = JSON.parse(row.specJson) as ArtifactSpec;
        const validated = parsed && parsed.sections?.length ? parsed : null;
        if (!validated) throw new Error("empty spec");
        spec = validated;
    } catch {
        await interaction
            .reply({ content: "O conteúdo guardado deste cartão está ilegível. Pede para editar ou reenviar o artefacto.", flags: MessageFlags.Ephemeral })
            .catch(() => undefined);
        return true;
    }

    const pageCount = spec.sections.length;
    let section = Math.min(Math.max(row.viewSection, 0), pageCount - 1);
    if (navMatch) {
        if (action === "prev") section = Math.max(0, section - 1);
        if (action === "next") section = Math.min(pageCount - 1, section + 1);
        if (action === "tab") {
            const rawValue = interaction.isStringSelectMenu() ? (interaction as StringSelectMenuInteraction).values?.[0] : undefined;
            const index = Number(rawValue);
            if (!Number.isFinite(index)) {
                await interaction.deferUpdate().catch(() => undefined);
                return true;
            }
            section = Math.min(Math.max(Math.floor(index), 0), pageCount - 1);
        }
        // "count" is the disabled page indicator; nothing to change.
        if (action === "count") {
            await interaction.deferUpdate().catch(() => undefined);
            return true;
        }

        // Re-render with a fresh nonce; the handler is central, so the customIds
        // are arbitrary as long as they match the pattern.
        const nonce = randomUUID().replace(/-/g, "").slice(0, 12);
        const built = buildArtifactComponents(spec, { section }, nonce);
        await ArtifactStore.updateViewState(interaction.message.id, section).catch(() => undefined);
        await interaction
            .update({ components: built.components, flags: MessageFlags.IsComponentsV2 })
            .catch(() => undefined);
        return true;
    }

    // Custom action rows and minigame buttons (prefix action: / game:)
    return handleCustomAction(interaction, spec, row);
}

async function handleCustomAction(
    interaction: ButtonInteraction | AnySelectMenuInteraction,
    spec: ArtifactSpec,
    row: { messageId: string; gameStateJson: string | null },
): Promise<boolean> {
    const customId = interaction.customId;

    let state: Record<string, unknown> = {};
    try {
        state = row.gameStateJson ? (JSON.parse(row.gameStateJson) as Record<string, unknown>) : { ...(spec.gameState ?? {}) };
    } catch {
        state = { ...(spec.gameState ?? {}) };
    }

    // Sandboxed script handlers: the model wrote JS at send time. Run it with
    // the persisted state and a bounded API (reply/send/render helpers).
    const handlerCode = spec.handlers?.[customId];
    if (handlerCode) {
        const scriptResult = runArtifactScript({
            code: handlerCode,
            state,
            user: { id: interaction.user.id, username: interaction.user.username },
            values: interaction.isAnySelectMenu() ? interaction.values ?? [] : [],
            customId,
            cardId: row.messageId,
        });

        if (scriptResult.error) {
            await interaction
                .reply({ content: `⚠️ Script error: ${scriptResult.error}`, flags: MessageFlags.Ephemeral })
                .catch(() => undefined);
            return true;
        }

        await ArtifactStore.updateGameState(row.messageId, scriptResult.state).catch(() => undefined);

        // Re-render when the script changed card-level state.
        const sectionChanged = scriptResult.section != null;
        const titleChanged = scriptResult.title != null;
        const accentChanged = scriptResult.accentColor != null;
        const spoilerChanged = scriptResult.spoiler != null;
        if (sectionChanged || titleChanged || accentChanged || spoilerChanged) {
            const nextSection = sectionChanged ? Math.max(0, Math.floor(scriptResult.section!)) : undefined;
            const updatedSpec: ArtifactSpec = {
                ...spec,
                ...(titleChanged ? { title: scriptResult.title! } : {}),
                ...(scriptResult.summary != null ? { summary: scriptResult.summary } : {}),
                ...(accentChanged ? { accentColor: scriptResult.accentColor! } : {}),
                ...(spoilerChanged ? { spoiler: scriptResult.spoiler! } : {}),
            };
            const nonce = randomUUID().replace(/-/g, "").slice(0, 12);
            const built = buildArtifactComponents(updatedSpec, { section: nextSection ?? 0 }, nonce);
            await interaction
                .update({ components: built.components, flags: MessageFlags.IsComponentsV2 })
                .catch(() => undefined);
        } else {
            await interaction.deferUpdate().catch(() => undefined);
        }

        // Deliver scripted sends (max 3) to their target channels.
        const channel = interaction.channel;
        for (const outgoing of scriptResult.sends) {
            try {
                const target = outgoing.channelId === (channel?.id ?? "") ? channel : await interaction.client.channels.fetch(outgoing.channelId);
                if (target && "send" in target) await (target as { send: (o: unknown) => Promise<unknown> }).send({ content: outgoing.content });
            } catch {
                // A failing send target must not break the card.
            }
        }

        const replyText = [scriptResult.reply, ...scriptResult.logs].filter(Boolean).join("\n");
        if (replyText) {
            await interaction
                .followUp({ content: replyText.slice(0, 1500), flags: MessageFlags.Ephemeral })
                .catch(() => undefined);
        }
        return true;
    }

    // Find the originating definition for a nicer ephemeral reply, if any.
    const label = findActionLabel(spec, customId);

    // Game state: any customId starting with "game:" mutates the persisted
    // JSON so minigames keep score across clicks and restarts. The FULL
    // suffix after "game:" is the state key, so `game:cell:0` and
    // `game:cell:1` are distinct counters — grids work without handlers.
    if (customId.startsWith("game:")) {
        const key = customId.slice(5).trim() || "score";
        const current = Number(state[key] ?? 0);
        state[key] = Number.isFinite(current) ? current + 1 : 1;
        state["_lastAction"] = customId;
        state["_lastUser"] = interaction.user.id;
        await ArtifactStore.updateGameState(row.messageId, state).catch(() => undefined);
        const prettyKey = key === "score" ? "score" : key;
        await interaction
            .reply({
                content: label ? `✅ ${label} (${key}): ${String(state[key])}` : `✅ ${prettyKey}: ${String(state[key])}`,
                flags: MessageFlags.Ephemeral,
            })
            .catch(() => undefined);
        return true;
    }

    // Plain action: buttons/selects with prefix action:
    if (customId.startsWith("action:")) {
        // Select values: show what was picked
        if (interaction.isStringSelectMenu() || interaction.isUserSelectMenu() || interaction.isRoleSelectMenu() || interaction.isMentionableSelectMenu() || interaction.isChannelSelectMenu()) {
            const values = (interaction as AnySelectMenuInteraction).values ?? [];
            await interaction
                .reply({
                    content: label ? `Selecionaste em **${label}**: ${values.join(", ") || "(vazio)"}` : `Selecionaste: ${values.join(", ") || "(vazio)"}`,
                    flags: MessageFlags.Ephemeral,
                })
                .catch(() => undefined);
            return true;
        }
        await interaction
            .reply({ content: label ? `Clicaste em **${label}**.` : "Interação recebida.", flags: MessageFlags.Ephemeral })
            .catch(() => undefined);
        return true;
    }

    // Fallback: unknown custom prefix, still acknowledge
    await interaction.deferUpdate().catch(() => undefined);
    return true;
}

function findActionLabel(spec: ArtifactSpec, customId: string): string | null {
    if (!spec.actionRows?.length) return null;
    for (const row of spec.actionRows) {
        if (row.type === "buttons") {
            for (const button of row.buttons) {
                if (button.customId === customId) return button.label;
            }
        }
        if ("customId" in row && (row as { customId?: string }).customId === customId) return (row as { customId?: string }).customId ?? null;
    }
    return null;
}
