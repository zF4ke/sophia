import { TextChannel, Message, ChatInputCommandInteraction } from "discord.js";
import { EMOJIS } from "../utils/constants";

export class MessageService {
    private static readonly BATCH_SIZE = 100;
    private static readonly MAX_EMPTY_FETCHES = 3;

    public static async fetchMessages(
        channel: TextChannel, 
        limit: number, 
        interaction: ChatInputCommandInteraction
    ): Promise<Message[]> {
        const messages: Message[] = [];
        let lastId: string | undefined;
        let consecutiveEmptyFetches = 0;
        let totalFetches = 0;
        const maxFetchAttempts = Math.ceil(limit / this.BATCH_SIZE) + this.MAX_EMPTY_FETCHES;

        while (messages.length < limit && consecutiveEmptyFetches < this.MAX_EMPTY_FETCHES && totalFetches < maxFetchAttempts) {
            try {
                const options: { limit: number; before?: string } = { 
                    limit: Math.min(this.BATCH_SIZE, limit - messages.length)
                };
                if (lastId) options.before = lastId;

                const fetchedMessages = await channel.messages.fetch(options);
                totalFetches++;

                if (!fetchedMessages || fetchedMessages.size === 0) {
                    consecutiveEmptyFetches++;
                    if (consecutiveEmptyFetches >= this.MAX_EMPTY_FETCHES) break;
                } else {
                    consecutiveEmptyFetches = 0;
                    const fetchedArray = [...fetchedMessages.values()];
                    messages.push(...fetchedArray);
                    lastId = fetchedArray[fetchedArray.length - 1]?.id;

                    if (messages.length % 1000 === 0) {
                        await interaction.editReply(
                            `${EMOJIS.search} Buscando mensagens... (${messages.length}/${limit} mensagens encontradas)`
                        );
                    }

                    if (totalFetches % 5 === 0) {
                        await new Promise(resolve => setTimeout(resolve, 1000));
                    }
                }
            } catch (error) {
                console.error('Error fetching messages:', error);
                if (error && typeof error === 'object' && 'code' in error && error.code === 50001) {
                    throw new Error('Não tenho permissão para ler mensagens neste canal.');
                }
                await new Promise(resolve => setTimeout(resolve, 2000));
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