import type {
    CategoryChannel,
    ChatInputCommandInteraction,
    TextChannel,
    ThreadChannel,
} from "discord.js";
import { DiscordMemoryService } from "@/services/memory/DiscordMemoryService";
import { UIService } from "@/services/UIService";
import { EMOJIS } from "@/utils/constants";

type IndexableChannel = TextChannel | ThreadChannel;

export class DiscordBackfillService {
    public static async backfillChannel(
        channel: IndexableChannel,
        interaction?: ChatInputCommandInteraction,
        limit = 1000
    ): Promise<number> {
        let fetched = 0;
        let before: string | undefined;

        while (fetched < limit) {
            const batch = await channel.messages.fetch({
                limit: Math.min(100, limit - fetched),
                before,
            });

            if (!batch.size) {
                break;
            }

            const messages = Array.from(batch.values()).sort(
                (a, b) => a.createdTimestamp - b.createdTimestamp
            );

            for (const message of messages) {
                await DiscordMemoryService.ingestMessage(message);
                fetched += 1;
            }

            before = messages[0]?.id;

            if (interaction) {
                await interaction.editReply(
                    UIService.formatStatusMessage(
                        EMOJIS.sync,
                        `Indexing ${channel.name}... ${fetched}/${limit}`
                    )
                );
            }
        }

        return fetched;
    }

    public static async backfillCategory(
        category: CategoryChannel,
        interaction?: ChatInputCommandInteraction,
        limitPerChannel = 500
    ): Promise<{ channels: number; messages: number }> {
        let channels = 0;
        let messages = 0;

        const children = category.children.cache.filter((channel) =>
            this.isIndexableChannel(channel)
        );
        for (const child of children.values()) {
            channels += 1;
            messages += await this.backfillChannel(child, interaction, limitPerChannel);
        }

        return { channels, messages };
    }

    private static isIndexableChannel(channel: unknown): channel is IndexableChannel {
        return Boolean(channel) && typeof channel === "object" && "messages" in (channel as object);
    }
}
