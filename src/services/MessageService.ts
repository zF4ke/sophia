import { TextChannel, Message, ChatInputCommandInteraction, Collection } from "discord.js";
import { EMOJIS } from "../utils/constants";
import { FileSystemService } from "./FileSystemService";
import { UIService } from "./UIService";

import path from "path";

interface MessageCache {
    messages: MessageCacheItem[];
    timestamp: number;
    lastAccessed?: number; // Track when the cache was last accessed
}

/**
 * Simplified message structure for cache storage
 * Contains only the essential data needed for cache comparison
 */
interface MessageCacheItem {
    id: string;
    channelId: string;
    authorId: string;
    authorUsername: string;
    content: string;
    createdTimestamp: number;
    reference?: { messageId?: string; };
}

export class MessageService {
    // Service name used for file organization
    private static readonly SERVICE_NAME = 'message';
    
    private static readonly MAX_BATCH_SIZE = 100; // Discord's max
    private static readonly MAX_TRIES = 3; // Max retries for fetching messages
    private static PAUSE_INTERVAL = 2000;
    private static MAX_CONSECUTIVE_REQUESTS = 20;
    private static readonly MIN_MESSAGES_FOR_CHANNEL_END = 25; 

    private static readonly USE_CACHE = true;
    private static readonly CACHE_EXPIRY = 30 * 24 * 60 * 60 * 1000; // 30 days
    private static readonly DEFAULT_LIMIT = 1000;
    private static readonly DEFAULT_CACHE_LIMIT = 10000; // read at most 10k messages from cache unless specified 
    private static readonly CACHE_MEMORY_TTL = 5 * 60 * 1000; // Time to keep cache in memory (5 minutes)
    private static readonly MAX_CACHE_SIZE = 50 * 1024 * 1024; // 50MB em bytes
    private static readonly CACHE_CLEANUP_THRESHOLD = 0.7; // Limpar até 70% do tamanho máximo
    private static readonly messageCache = new Map<string, MessageCache>();

    /**
     * Loads cache for a specific channel from disk if it exists and is valid
     * @param channelId The channel ID to load cache for
     * @returns The loaded cache data or null if no valid cache exists
     */
    private static loadChannelCache(channelId: string): MessageCache | null {
        if (!this.USE_CACHE) return null;

        try {
            // Check if we already have it in memory and it's not expired
            const existingCache = this.messageCache.get(channelId);
            if (existingCache) {
                existingCache.lastAccessed = Date.now();
                return existingCache;
            }

            // Try to load from disk
            const cacheData = FileSystemService.readJsonFromPath<MessageCache>(`${channelId}.json`, this.SERVICE_NAME, 'cache');
            
            if (cacheData && cacheData.messages && cacheData.timestamp) {
                // Check if cache is still valid
                if (Date.now() - cacheData.timestamp < this.CACHE_EXPIRY) {
                    cacheData.lastAccessed = Date.now();
                    this.messageCache.set(channelId, cacheData);
                    return cacheData;
                } else {
                    // Delete expired cache file
                    FileSystemService.deleteFileFromPath(`${channelId}.json`, this.SERVICE_NAME, 'cache');
                }
            }
        } catch (err) {
            console.error(`Error loading cache for channel ${channelId}:`, err);
        }

        return null;
    }

    /**
     * Cleanup old caches from memory
     */
    private static cleanupMemoryCache(): void {
        const now = Date.now();
        for (const [channelId, cache] of this.messageCache.entries()) {
            if (cache.lastAccessed && (now - cache.lastAccessed) > this.CACHE_MEMORY_TTL) {
                this.messageCache.delete(channelId);
            }
        }
    }

    /**
     * Verifica e limpa o cache se necessário para manter abaixo do limite máximo
     */
    private static async cleanupCacheIfNeeded(): Promise<void> {
        const cacheDir = FileSystemService.getDir(this.SERVICE_NAME, 'cache');
        const currentSize = FileSystemService.getDirectorySize(cacheDir);

        // Se o tamanho atual é menor que o máximo, não precisa limpar
        if (currentSize <= this.MAX_CACHE_SIZE) {
            return;
        }

        // Calcular o tamanho alvo (70% do máximo)
        const targetSize = this.MAX_CACHE_SIZE * this.CACHE_CLEANUP_THRESHOLD;

        // Obter lista de arquivos do cache ordenados por data de criação (mais antigos primeiro)
        const files = FileSystemService.getDirectoryFiles(cacheDir)
            .sort((a, b) => a.created - b.created);

        let currentTotalSize = currentSize;

        // Remover arquivos mais antigos até atingir o tamanho alvo
        for (const file of files) {
            if (currentTotalSize <= targetSize) {
                break;
            }

            // Remover arquivo do disco e da memória
            const channelId = path.basename(file.path, '.json');
            //without using path.basename
            
            console.log(`Removendo cache de ${channelId} (${file.size} bytes) path: ${file.path}`);
            
            this.messageCache.delete(channelId);
            FileSystemService.deleteFile(file.path);

            // check if the file was deleted successfully
            if (FileSystemService.fileExists(file.path)) {
                console.error(`Falha ao remover o cache de ${channelId}`);
                continue;
            } else {
                console.log(`Cache de ${channelId} removido com sucesso`);
            }

            currentTotalSize -= file.size;
        }
    }

    /**
     * Verifica se adicionar novos dados excederá o limite de cache
     * @param newData Dados a serem adicionados
     * @returns true se exceder o limite
     */
    private static willExceedCacheLimit(newData: any): boolean {
        const newSize = Buffer.from(JSON.stringify(newData)).length;
        const cacheDir = FileSystemService.getDir(this.SERVICE_NAME, 'cache');
        const currentSize = FileSystemService.getDirectorySize(cacheDir);

        return (currentSize + newSize) > this.MAX_CACHE_SIZE;
    }

    /**
     * Saves the cache to disk for persistence
     * @param channelId The channel ID to save cache for
     */
    private static saveCache(channelId: string): void {
        if (!this.USE_CACHE) return;

        try {
            let cacheData = this.messageCache.get(channelId);
            if (!cacheData) return;

            // Verificar se vai exceder o limite antes de salvar
            if (this.willExceedCacheLimit(cacheData)) {
                this.cleanupCacheIfNeeded();
                cacheData = this.messageCache.get(channelId);
                if (!cacheData) return; // Re-check after cleanup
            }

            // Use FileSystemService to write to the cache directory
            FileSystemService.writeJsonToPath(`${channelId}.json`, cacheData, this.SERVICE_NAME, 'cache');
        } catch (err) {
            console.error(`Error saving cache for channel ${channelId}:`, err);
        }
    }

    /**
     * Converts a Discord Message to a simplified cache item
     * @param message Discord Message object
     * @returns Simplified message cache item
     */
    private static messageToCache(message: Message): MessageCacheItem {
        return {
            id: message.id,
            channelId: message.channelId,
            authorId: message.author.id,
            authorUsername: message.author.username,
            content: message.content,
            createdTimestamp: message.createdTimestamp,
            reference: message.reference ? { messageId: message.reference.messageId } : undefined
        };
    }

    /**
     * Gets the size of the cache for a specific channel
     * @param channelId The channel ID to check
     * @returns The number of messages in cache for this channel, or 0 if no cache exists
     */
    public static getCacheSize(channelId: string): number {
        const cachedData = this.loadChannelCache(channelId);
        if (!cachedData || (Date.now() - cachedData.timestamp) >= this.CACHE_EXPIRY) {
            return 0;
        }
        return cachedData.messages.length;
    }

    private static async updateStatus(
        interaction: ChatInputCommandInteraction | undefined, 
        message: string,
        emoji: string = EMOJIS.search,
        useBackticks: boolean = true
    ): Promise<void> {
        if (interaction) {
            await interaction.editReply(UIService.formatStatusMessage(emoji, message, useBackticks));
        } else {
            // console.log(`${emoji} ${message}`);
        }
    }

    /**
     * Fetches messages from a Discord channel with progress updates
     * Uses smart caching to minimize API calls
     * @param channel The Discord text channel to fetch messages from
     * @param limit Maximum number of messages to fetch
     * @param interaction Discord interaction for progress updates
     * @returns Array of Discord messages
     */
    public static async fetchMessages(
        channel: TextChannel, 
        limit: number, 
        interaction?: ChatInputCommandInteraction,
        useCache: boolean = this.USE_CACHE
    ): Promise<Message[]> {        
        let messages: Message[] = [];
        
        if (useCache) {
            messages = await this.fetchMessagesWithCache(channel, limit, interaction);
        } else {
            messages = await this.fetchMessagesWithoutCache(channel, limit, interaction);
        }

        if (messages.length === 0) {
            await this.updateStatus(interaction, "Nenhuma mensagem encontrada no canal.", EMOJIS.warning, false);
            return [];
        }

        return this.sortMessages(messages, true);
    }

    /**
     * Fetches messages from a Discord channel without using cache
     * @param channel The Discord text channel to fetch messages from
     * @param limit Maximum number of messages to fetch
     * @param interaction Discord interaction for progress updates
     * @returns Array of Discord messages
     */
    private static async fetchMessagesWithoutCache(
        channel: TextChannel,
        limit: number,
        interaction?: ChatInputCommandInteraction,
        options?: {
            lastMessageId?: string;
        }
    ): Promise<Message[]> {
        const channelId = channel.id;
        const amountOfMessagesToFetch = limit && limit != 0 ? limit : this.DEFAULT_LIMIT;
        const batchSize = Math.min(this.MAX_BATCH_SIZE, amountOfMessagesToFetch);
        const amountOfBatches = Math.ceil(limit / batchSize);

        let messages: Message[] = [];
        let lastMessageId: string | undefined = options?.lastMessageId;
        let messagesLeft = amountOfMessagesToFetch;
        let batchCountSincePause = 0;

        let batchCount = 0;

        // fetch all batches
        while (messagesLeft > 0) {
            if (messagesLeft <= 0) break; // No more messages to fetch

            if (batchCount % 5 === 0) {
                await this.updateStatus(interaction, `Buscando mensagens... (${messages.length}/${amountOfMessagesToFetch})`, EMOJIS.search, true);
            }

            const size = Math.min(batchSize, messagesLeft);
            const { 
                messages: fetchedMessages, 
                numberFetches, 
            } = await this.fetchBatch(interaction, channel, size, lastMessageId, batchCountSincePause);
            batchCountSincePause = numberFetches;

            messages.push(...fetchedMessages);

            // remove duplicates
            const uniqueMessages = new Map(messages.map(msg => [msg.id, msg]));
            messages = Array.from(uniqueMessages.values());

            lastMessageId = fetchedMessages.length > 0 ? fetchedMessages[fetchedMessages.length - 1].id : undefined;
            messagesLeft -= fetchedMessages.length;

            if (fetchedMessages && fetchedMessages.length > 0 && fetchedMessages.length < size) {
                // If less messages were fetched than requested, we reached the end of the channel
                messagesLeft = 0; // No more messages to fetch
                break;
            }
            if (messagesLeft <= 0) break; // No more messages to fetch

            batchCount++;
        }

        // update cache with new messages
        this.updateCache(channelId, messages);

        const orderedMessages = this.sortMessages(messages); 
        messages = orderedMessages.slice(0, amountOfMessagesToFetch); // Limit to the requested amount

        return messages;
    }

    /**
     * Fetches a batch of messages from a Discord channel
     * @param channel The Discord text channel to fetch messages from
     * @param limit Maximum number of messages to fetch in this batch
     * @param lastMessageId ID of the last message fetched in the previous batch (for pagination)
     * @param batchCount Current batch count for progress updates
     * @returns Array of Discord messages
     */
    private static async fetchBatch(
        interaction: ChatInputCommandInteraction | undefined,
        channel: TextChannel,
        limit: number,
        lastMessageId?: string,
        batchCountSincePause: number = 0,
        tries: number = 0,
    ): Promise<{ messages: Message[]; numberFetches: number }> {
        try {
            if (tries >= this.MAX_TRIES) {
                await this.updateStatus(interaction, "Excedeu o número máximo de tentativas para buscar mensagens.", EMOJIS.error, false);
            }

            // check rate limit
            if (batchCountSincePause >= this.MAX_CONSECUTIVE_REQUESTS) {
                // console.log(`Atingido o limite de requisições consecutivas. Aguardando ${this.PAUSE_INTERVAL/1000}s...`);
                //await this.updateStatus(interaction, `Aguardando ${this.PAUSE_INTERVAL/1000}s...`, EMOJIS.network, true);
                await new Promise(resolve => setTimeout(resolve, this.PAUSE_INTERVAL));
                batchCountSincePause = 0; // Reset the counter after waiting
            }

            // fetch messages from Discord
            const fetchedMessages = await channel.messages.fetch({
                limit: limit,
                before: lastMessageId,
            });

            batchCountSincePause++;
    
            return {
                messages: Array.from(fetchedMessages.values()),
                numberFetches: batchCountSincePause,
            }
        } catch (error: any) {
            // check if the error is a rate limit error
            if (error?.code === 429) {
                const retryAfter = error.retry_after * 1000 || 1000;
                await this.updateStatus(
                    interaction,
                    `Limite de API atingido. Aguardando ${retryAfter/1000}s...`,
                    EMOJIS.network,
                    true
                );
                //console.log(`Limite de API atingido depois de ${batchCountSincePause} requisições. Aguardando ${retryAfter}ms...`);
                await new Promise(resolve => setTimeout(resolve, retryAfter));

                // update MAX_CONSECUTIVE_REQUESTS to avoid hitting the rate limit again
                this.MAX_CONSECUTIVE_REQUESTS--;
                // increase the pause interval by 50%
                this.PAUSE_INTERVAL *= 1.5;
                batchCountSincePause = 0; // Reset the counter after waiting

                return this.fetchBatch(interaction, channel, limit, lastMessageId, batchCountSincePause, tries + 1);
            }
        }

        return { messages: [], numberFetches: batchCountSincePause };
    }

    /**
     * Fetches messages from a Discord channel using cache if available
     * @param channel The Discord text channel to fetch messages from
     * @param limit Maximum number of messages to fetch
     * @param interaction Discord interaction for progress updates
     * @returns Array of Discord messages
     */
    private static async fetchMessagesWithCache(
        channel: TextChannel,
        limit: number,
        interaction?: ChatInputCommandInteraction,
    ): Promise<Message[]> {
        const channelId = channel.id;
        const amountOfMessagesToFetch = limit && limit != 0 ? limit : Math.min(this.getCacheSize(channelId), this.DEFAULT_CACHE_LIMIT);
        const batchSize = Math.min(this.MAX_BATCH_SIZE, amountOfMessagesToFetch);
        const amountOfBatches = Math.ceil(limit / batchSize);

        let messages: Message[] = [];
        let lastMessageId: string | undefined = undefined;
        let messagesLeft = amountOfMessagesToFetch;
        let batchCountSincePause = 0;

        const cachedData = this.loadChannelCache(channelId);
        if (!cachedData) {
            await this.updateStatus(interaction, "Nenhum cache encontrado. Buscando mensagens do Discord...", EMOJIS.cache, true);
            const amount = limit && limit != 0 ? limit : this.DEFAULT_LIMIT;
            // No cache available, fetch messages from Discord
            return this.fetchMessagesWithoutCache(channel, amount, interaction);
        }

        // Check if cache is expired
        if (Date.now() - cachedData.timestamp >= this.CACHE_EXPIRY) {
            await this.updateStatus(interaction, "Cache expirado. Buscando mensagens do Discord...", EMOJIS.cache, true);
            const amount = limit && limit != 0 ? limit : this.DEFAULT_LIMIT;
            // Cache expired, fetch messages from Discord
            return this.fetchMessagesWithoutCache(channel, amount, interaction);
        }

        // Cache is valid, use it
        const messagesOnCache = this.reconstructMessages(channel, cachedData.messages);

        let cacheHit = false;
        let batchCount = 0;

        // Start fetching fresh batches from Discord until we find matching messages that are in cache
        while (messagesLeft > 0) {
            if (messagesLeft <= 0) break; // No more messages to fetch

            if (batchCount % 5 === 0) {
                await this.updateStatus(interaction, `Buscando mensagens... (${messages.length}/${amountOfMessagesToFetch})`, EMOJIS.search, true);
            }

            const size = Math.min(batchSize, messagesLeft);
            const { 
                messages: fetchedMessages, 
                numberFetches, 
            } = await this.fetchBatch(interaction, channel, size, lastMessageId, batchCountSincePause);
            batchCountSincePause = numberFetches;

            messages.push(...fetchedMessages);
            lastMessageId = fetchedMessages.length > 0 ? fetchedMessages[fetchedMessages.length - 1].id : undefined;
            messagesLeft -= fetchedMessages.length;

            if (fetchedMessages && fetchedMessages.length > 0 && fetchedMessages.length < size) {
                // If less messages were fetched than requested, we reached the end of the channel
                messagesLeft = 0; // No more messages to fetch
                break;
            }
            if (messagesLeft <= 0) break; // No more messages to fetch

            if (this.matchedCacheMessages(messagesOnCache, fetchedMessages)) {
                // We found matching messages in cache, stop fetching
                cacheHit = true;
                break;
            }

            batchCount++;
        }

        if (cacheHit) {
            // We found matching messages in cache, merge them with fetched messages, prioritizing discord messages
            const cacheMessages = messagesOnCache.filter(msg => !messages.some(fetchedMsg => fetchedMsg.id === msg.id));
            messages = [...messages, ...cacheMessages];

            messages = this.sortMessages(messages); // Sort by creation time

            // remove duplicates
            const uniqueMessages = new Map(messages.map(msg => [msg.id, msg]));
            messages = Array.from(uniqueMessages.values());

            await this.updateStatus(interaction, `Encontradas ${messages.length} mensagens no cache.`, EMOJIS.cache, true);
            // update cache timestamp
            cachedData.timestamp = Date.now();
            cachedData.lastAccessed = Date.now();
        }

        messagesLeft = amountOfMessagesToFetch - messages.length;

        // check if we have enough messages
        if (amountOfMessagesToFetch <= messages.length) {
            messages = messages.slice(0, amountOfMessagesToFetch);
            return messages;
        }
        
        await this.updateStatus(interaction, `Buscando ${messagesLeft} mensagens restantes...`, EMOJIS.search, true);

        // We need to fetch more messages from Discord
        lastMessageId = messages[messages.length - 1].id; // Get the last message ID from cache

        const restOfMessages = await this.fetchMessagesWithoutCache(channel, messagesLeft, interaction, { lastMessageId });
        messages.push(...restOfMessages);

        const orderedMessages = this.sortMessages(messages); // Sort by creation time (oldest first)
        messages = orderedMessages.slice(0, amountOfMessagesToFetch); // Limit to the requested amount

        return messages;
    }

    /**
     * Checks if there are matching messages in cache
     * @param cachedMessages Messages from cache
     * @param fetchedMessages Messages fetched from Discord
     * @returns true if there are matching messages, false otherwise
     */
    private static matchedCacheMessages(cachedMessages: Message[], fetchedMessages: Message[]): boolean {
        const cachedIds = new Set(cachedMessages.map(msg => msg.id));
        for (const message of fetchedMessages) {
            if (cachedIds.has(message.id)) {
                return true; // Found a matching message in cache
            }
        }
        return false; // No matching messages found
    }

    /**
     * Sorts messages by creation time
     * @param messages Array of Discord messages to sort
     * @param oldestFirst If true, sorts from oldest to newest. Otherwise, sorts from newest to oldest.
     * @returns Sorted array of Discord messages
     */
    public static sortMessages(
        messages: Message[],
        oldestFirst: boolean = false
    ): Message[] {
        if (oldestFirst) {
            return messages.sort((a, b) => a.createdTimestamp - b.createdTimestamp);
        } else {
            return messages.sort((a, b) => b.createdTimestamp - a.createdTimestamp);
        }
    }

    /**
     * Fetches messages from a Discord channel in a specific order (oldest first)
     * @param channel The Discord text channel to fetch messages from
     * @param limit Maximum number of messages to fetch
     * @param interaction Discord interaction for progress updates
     * @returns Array of Discord messages in the specified order
     */
    public static async fetchMessagesOrdered(
        channel: TextChannel,
        limit: number,
        interaction: ChatInputCommandInteraction,
    ): Promise<Message[]> {
        // fetch 100 messages, it's the maximum Discord allows
        const messages = await channel.messages.fetch({ limit: this.MAX_BATCH_SIZE });
        if (messages.size === 0) {
            await interaction.editReply(
                UIService.formatStatusMessage(EMOJIS.warning, "Nenhuma mensagem encontrada no canal.", false)
            );
            return [];
        }

        // Sort messages by creation time (newest first)
        const sortedMessages = messages.sort((a, b) => a.createdTimestamp - b.createdTimestamp);

        // convert to Message[]

        const messageArray = Array.from(sortedMessages.values());
        if (messageArray.length > limit) {
            messageArray.length = limit; // truncate to limit
        }

        return messageArray;
    }

    /**
     * Updates the message cache with new messages and saves to disk
     * @param channelId Channel ID
     * @param messages New messages to add to cache
     */
    private static updateCache(channelId: string, messages: Message[]): void {
        if (!this.USE_CACHE || messages.length === 0) return;

        try {
            // Convert messages to cache format
            const newCacheMessages = messages.map(msg => this.messageToCache(msg));
            
            // Get existing cache or create new one
            const existing = this.loadChannelCache(channelId);
            if (existing) {
                // Create a set of existing IDs for fast lookup
                const existingIds = new Set(existing.messages.map(msg => msg.id));
                
                // Add only new messages to cache
                for (const message of newCacheMessages) {
                    if (!existingIds.has(message.id)) {
                        existing.messages.push(message);
                    }
                }
                
                // Update timestamp and last accessed
                existing.timestamp = Date.now();
                existing.lastAccessed = Date.now();
                this.messageCache.set(channelId, existing);
            } else {
                // Create new cache entry
                const newCache: MessageCache = {
                    messages: newCacheMessages,
                    timestamp: Date.now(),
                    lastAccessed: Date.now()
                };
                this.messageCache.set(channelId, newCache);
            }
            
            // Save to disk
            this.saveCache(channelId);
        } catch (err) {
            console.error(`Error updating cache for channel ${channelId}:`, err);
        }
    }

    /**
     * Reconstructs Discord Message objects from cached data
     * @param channel The channel the messages belong to
     * @param cachedMessages Array of cached message data
     * @returns Array of reconstructed Message-like objects
     */
    private static reconstructMessages(channel: TextChannel, cachedMessages: MessageCacheItem[]): Message[] {
        // Create simplified Message-like objects from cache data
        // These won't have all Message methods but will work for most use cases
        return cachedMessages.map(cached => {
            const partial = {
                id: cached.id,
                channelId: cached.channelId,
                channel: channel,
                author: {
                    id: cached.authorId,
                    username: cached.authorUsername
                },
                content: cached.content,
                createdTimestamp: cached.createdTimestamp,
                reference: cached.reference,
                guild: channel.guild
            } as unknown as Message;
            
            return partial;
        });
    }

    public static filterCommandMessages(
        messages: Message[], 
        interaction: ChatInputCommandInteraction
    ): Message[] {
        return messages.filter(msg => {
            // Filter out bot's own messages
            if (msg.author.id === interaction.client.user?.id) return false;

            // Filter out recent command messages from the command author
            if (msg.author.id === interaction.user.id) {
                const timeDiff = interaction.createdTimestamp - msg.createdTimestamp;
                if (Math.abs(timeDiff) < 10000) return false;
            }
            return true;
        });
    }

    /**
     * * Filters out messages that are from the bot itself or are bot messages.
     * @param messages - The messages to filter.
     * * @param botId - The bot's ID to filter out its own messages.
     * @returns An array of messages that are not from the bot or are not bot messages.
     */
    public static filterOwnMessages(messages: Message[], botId: string): Message[] {
        return messages.filter(message => {
            return message.author.id !== botId;
        });
    }

    /**
     * Clears the message cache for a specific channel or all channels
     * @param channelId Optional channel ID to clear cache for. If not provided, clears all message caches.
     */
    public static clearCache(channelId?: string): void {
        if (channelId) {
            this.messageCache.delete(channelId);
            FileSystemService.deleteFileFromPath(`${channelId}.json`, this.SERVICE_NAME, 'cache');
        } else {
            this.messageCache.clear();
            FileSystemService.clearDirectory('.json', this.SERVICE_NAME, 'cache');
        }
    }
}