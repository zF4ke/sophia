import { TextChannel, Message, ChatInputCommandInteraction, Collection } from "discord.js";
import { EMOJIS } from "../utils/constants";

export class MessageService {
    private static readonly MAX_BATCH_SIZE = 100; // Discord's max
    private static readonly UPDATE_INTERVAL = 5000;
    private static readonly MIN_MESSAGES_FOR_CHANNEL_END = 25; // If we get less than this, we're near channel end

    public static async fetchMessages(
        channel: TextChannel, 
        limit: number, 
        interaction: ChatInputCommandInteraction
    ): Promise<Message[]> {
        const messages: Message[] = [];
        let lastId: string | undefined;
        let lastProgressUpdate = Date.now();
        let lastBatchSize = this.MAX_BATCH_SIZE;

        while (messages.length < limit) {
            // Check if we're likely at channel end based on last batch size
            if (lastBatchSize < this.MIN_MESSAGES_FOR_CHANNEL_END && messages.length > 0) {
                break;
            }

            try {
                const fetchLimit = Math.min(this.MAX_BATCH_SIZE, limit - messages.length);
                const response = await channel.messages.fetch({ 
                    limit: fetchLimit,
                    ...(lastId && { before: lastId })
                });

                // Store batch size for early exit detection
                lastBatchSize = response.size;
                if (response.size === 0) break;

                messages.push(...response.values());
                lastId = response.last()?.id;

                if (Date.now() - lastProgressUpdate > this.UPDATE_INTERVAL) {
                    await interaction.editReply(
                        `${EMOJIS.search} Carregando mensagens... (${messages.length}/${limit})`
                    );
                    lastProgressUpdate = Date.now();
                }

            } catch (error: any) {
                if (error?.code === 50001) {
                    throw new Error('Não tenho permissão para ler mensagens neste canal.');
                }

                if (error?.code === 429) {
                    const retryAfter = error.retry_after * 1000 || 1000;
                    await new Promise(resolve => setTimeout(resolve, retryAfter));
                    continue;
                }

                // For non-rate-limit errors, wait a short time and retry once
                await new Promise(resolve => setTimeout(resolve, 100));
                try {
                    const retryResponse = await channel.messages.fetch({ 
                        limit: Math.min(this.MAX_BATCH_SIZE, limit - messages.length),
                        ...(lastId && { before: lastId })
                    });
                    lastBatchSize = retryResponse.size;
                    if (retryResponse.size > 0) {
                        messages.push(...retryResponse.values());
                        lastId = retryResponse.last()?.id;
                    }
                } catch {
                    console.error('Failed retry, continuing with next batch');
                    lastBatchSize = 0;
                }
            }
        }

        return messages;
    }

    public static filterCommandMessages(
        messages: Message[], 
        interaction: ChatInputCommandInteraction
    ): Message[] {
        return messages.filter(msg => {
            if (msg.author.id === interaction.user.id) {
                const timeDiff = interaction.createdTimestamp - msg.createdTimestamp;
                if (Math.abs(timeDiff) < 10000) return false;
            }
            return true;
        });
    }
}