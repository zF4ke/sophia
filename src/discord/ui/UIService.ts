import {
    ActionRowBuilder,
    ButtonBuilder,
    ButtonStyle,
    EmbedBuilder,
    Message,
    type ChatInputCommandInteraction,
} from "discord.js";
import { AnswerFormatter } from "@/discord/ui/formatters/AnswerFormatter";
import { MessageGroupFormatter } from "@/discord/ui/formatters/MessageGroupFormatter";
import { StatusFormatter } from "@/discord/ui/formatters/StatusFormatter";
import { NavigationCollector } from "@/discord/ui/interactions/NavigationCollector";
import { ChannelMessenger } from "@/discord/ui/transport/ChannelMessenger";
import { InteractionMessenger } from "@/discord/ui/transport/InteractionMessenger";
import type { NavigationButton, NavigationState } from "@/discord/ui/types";
import type { AnswerCitation } from "@/shared/appTypes";

export { NavigationCollector };
export type { NavigationButton, NavigationState };

export class UIService {
    public static readonly DEFAULT_COLOR = 0x7289da;

    public static formatStatusMessage(
        emoji: string,
        message: string,
        useBackticks = true
    ): string {
        return StatusFormatter.format(emoji, message, useBackticks);
    }

    public static formatAnswer(answer: string, citations: AnswerCitation[] = []) {
        return AnswerFormatter.format(answer, citations);
    }

    public static async sendLongResponse(
        interaction: ChatInputCommandInteraction,
        messageHeader: string,
        response: string,
        ephemeral = false
    ): Promise<void> {
        await InteractionMessenger.sendLongResponse(
            interaction,
            messageHeader,
            response,
            ephemeral
        );
    }

    public static async sendLongMessage(message: Message, response: string): Promise<void> {
        await ChannelMessenger.sendLongMessage(message, response);
    }

    protected static createNavigationRow(
        buttons: NavigationButton[]
    ): ActionRowBuilder<ButtonBuilder> {
        return new ActionRowBuilder<ButtonBuilder>().addComponents(
            ...buttons.map((button) =>
                new ButtonBuilder()
                    .setCustomId(button.customId)
                    .setLabel(button.label)
                    .setStyle(ButtonStyle[button.style])
                    .setDisabled(button.disabled)
            )
        );
    }

    protected static addMessageGroupFields(embed: EmbedBuilder, messages: Message[]): void {
        MessageGroupFormatter.addMessageGroupFields(embed, messages);
    }

    protected static async setupInteractionCollector<T>(
        message: Message,
        interaction: ChatInputCommandInteraction,
        items: T[],
        state: NavigationState,
        createEmbed: (state: NavigationState) => EmbedBuilder,
        createRow: (state: NavigationState) => ActionRowBuilder<ButtonBuilder>
    ): Promise<void> {
        await NavigationCollector.collect(
            message,
            interaction,
            items,
            state,
            createEmbed,
            createRow
        );
    }
}
