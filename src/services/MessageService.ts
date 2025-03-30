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
    private static readonly UPDATE_INTERVAL = 5000;
    private static readonly MIN_MESSAGES_FOR_CHANNEL_END = 25; // If we get less than this, we're near channel end

    private static readonly USE_CACHE = true; // Use cache to avoid hitting API limits
    private static readonly CACHE_EXPIRY = 30 * 24 * 60 * 60 * 1000; // 30 days cache expiry
    private static readonly DEFAULT_MAX_LIMIT = 1000; // Default maximum limit
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
        interaction: ChatInputCommandInteraction
    ): Promise<Message[]> {
        const channelId = channel.id;
        let messages: Message[] = [];
        let lastId: string | undefined;
        let lastProgressUpdate = Date.now();
        let lastBatchSize = this.MAX_BATCH_SIZE;
        let cacheChecksEnabled = this.USE_CACHE;
        
        // Load channel cache if needed
        let cachedData = cacheChecksEnabled ? this.loadChannelCache(channelId) : null;

        // Clean up old caches from memory periodically
        this.cleanupMemoryCache();
        
        // Adjust limit based on cache size if not explicitly provided
        // Use cache size if it's available and within DEFAULT_MAX_LIMIT (1000)
        if (limit === 0 && cachedData && cachedData.messages.length > 0) {
            limit = Math.min(cachedData.messages.length, this.DEFAULT_MAX_LIMIT);
            await interaction.editReply(
                UIService.formatStatusMessage(
                    EMOJIS.cache,
                    `Cache encontrado! Usando ${limit} mensagens do cache como limite.`
                )
            );
        } else if (limit === 0) {
            // If no cache and no explicit limit, use the default max limit
            limit = this.DEFAULT_MAX_LIMIT;
        }
        
        // Check if we have a valid cache to use
        const hasCachedData = cachedData && (Date.now() - cachedData.timestamp) < this.CACHE_EXPIRY;

        // Initialize cache lookup for repeated checks
        const cachedIds = hasCachedData ? new Set(cachedData.messages.map(msg => msg.id)) : new Set();
        const cachedMessagesMap = hasCachedData ? new Map(
            cachedData.messages.map(msg => [msg.id, msg])
        ) : new Map();

        // Try to use cache for initial fetch if enabled
        if (hasCachedData) {
            await interaction.editReply(
                UIService.formatStatusMessage(EMOJIS.cache, `Cache encontrado com ${cachedData.messages.length} mensagens. Verificando mensagens novas...`)
            );
            
            // First fetch newest messages and see if they match cache
            try {
                const newestBatch = await channel.messages.fetch({ 
                    limit: Math.min(this.MAX_BATCH_SIZE, limit) 
                });
                
                // Check if we've found any messages that are already cached
                const newestMessages = Array.from(newestBatch.values());
                
                // Add new messages to our results
                const newMessages = newestMessages.filter(msg => !cachedIds.has(msg.id));
                messages.push(...newMessages);
                
                // If we found cached messages in this batch, we can start using the cache
                const overlappingMessages = newestMessages.filter(msg => cachedIds.has(msg.id));
                if (overlappingMessages.length > 0) {
                    // We found an overlap, so we can use cache for older messages
                    const oldestNewMessageTimestamp = Math.min(...overlappingMessages.map(m => m.createdTimestamp));
                    
                    // Get older cached messages up to the limit
                    const olderCachedMessages = cachedData.messages
                        .filter(msg => msg.createdTimestamp < oldestNewMessageTimestamp)
                        .slice(0, limit - messages.length);
                    
                    // Add reconstructed messages to our results
                    if (olderCachedMessages.length > 0) {
                        const reconstructedOldMessages = this.reconstructMessages(channel, olderCachedMessages);
                        messages.push(...reconstructedOldMessages);
                        
                        await interaction.editReply(
                            UIService.formatStatusMessage(
                                EMOJIS.merge,
                                `Combinando ${newMessages.length} mensagens novas com ${olderCachedMessages.length} mensagens em cache`
                            )
                        );
                    }
                    
                    // If we have enough messages, we're done
                    if (messages.length >= limit) {
                        // Sort and return the messages
                        messages.sort((a, b) => b.createdTimestamp - a.createdTimestamp);
                        this.updateCache(channelId, messages);
                        return messages.slice(0, limit);
                    }
                }
                
                // Set lastId for continued fetching
                if (newestBatch.size > 0) {
                    lastId = newestBatch.last()?.id;
                }
                
            } catch (error) {
                console.error('Error fetching newest messages:', error);
                // If we can't fetch new messages, try using cache directly
                if (cachedData.messages.length >= limit) {
                    await interaction.editReply(
                        UIService.formatStatusMessage(
                            EMOJIS.cache,
                            `Usando mensagens em cache (${Math.min(limit, cachedData.messages.length)}/${limit})`
                        )
                    );
                    return this.reconstructMessages(channel, cachedData.messages.slice(0, limit));
                } else {
                    // Fallback to using whatever we have in cache and continue fetching
                    await interaction.editReply(
                        UIService.formatStatusMessage(
                            EMOJIS.cache,
                            `Reconstruindo ${cachedData.messages.length} mensagens a partir do cache...`
                        )
                    );
                    messages = this.reconstructMessages(channel, cachedData.messages);
                }
            }
        } else if (cacheChecksEnabled) {
            // Inform the user that no cache was found
            await interaction.editReply(
                UIService.formatStatusMessage(EMOJIS.cache, `Nenhum cache encontrado para este canal. Buscando do zero...`)
            );
        }

        // Fetch remaining messages if needed
        while (messages.length < limit) {
            // Check if we're likely at channel end based on last batch size
            if (lastBatchSize < this.MIN_MESSAGES_FOR_CHANNEL_END && messages.length > 0) {
                break;
            }

            try {
                const fetchLimit = Math.min(this.MAX_BATCH_SIZE, limit - messages.length);
                const response = await channel.messages.fetch({ 
                    limit: fetchLimit,
                    ...(lastId && { before: lastId }),
                });

                // Store batch size for early exit detection
                lastBatchSize = response.size;
                if (response.size === 0) break;

                // Get messages from this batch
                const batchMessages = Array.from(response.values());
                
                // Check for cached messages in this batch, if we have valid cache
                if (hasCachedData && batchMessages.length > 0) {
                    // Check if any of these messages are in our cache
                    const overlappingMessages = batchMessages.filter(msg => cachedIds.has(msg.id));
                    
                    if (overlappingMessages.length > 0) {
                        // We found more overlap! Let's use cache for any older messages we need
                        const oldestOverlapTimestamp = Math.min(...overlappingMessages.map(m => m.createdTimestamp));
                        
                        // Add only non-cached messages from this batch
                        const newMessages = batchMessages.filter(msg => !cachedIds.has(msg.id));
                        messages.push(...newMessages);
                        
                        // Find cached messages older than our overlap point
                        const olderCachedMessages = cachedData.messages
                            .filter(msg => msg.createdTimestamp < oldestOverlapTimestamp)
                            .slice(0, limit - messages.length);
                        
                        if (olderCachedMessages.length > 0) {
                            // We found older messages in cache, add them
                            const reconstructedOldMessages = this.reconstructMessages(channel, olderCachedMessages);
                            messages.push(...reconstructedOldMessages);
                            
                            await interaction.editReply(
                                UIService.formatStatusMessage(
                                    EMOJIS.found,
                                    `Encontradas ${reconstructedOldMessages.length} mensagens no cache! Total: ${messages.length}`
                                )
                            );
                            
                            // If we have enough messages now, we're done
                            if (messages.length >= limit) {
                                // Sort and return the messages
                                messages.sort((a, b) => b.createdTimestamp - a.createdTimestamp);
                                this.updateCache(channelId, messages);
                                return messages.slice(0, limit);
                            }
                            
                            // Update lastId to continue after the oldest cached message
                            if (olderCachedMessages.length > 0) {
                                const oldestCachedMessage = olderCachedMessages[olderCachedMessages.length - 1];
                                lastId = oldestCachedMessage.id;
                                continue;
                            }
                        }
                    } else {
                        // No overlap in this batch, add them all and continue
                        messages.push(...batchMessages);
                    }
                } else {
                    // No cache check needed, just add all messages
                    messages.push(...batchMessages);
                }

                // Update lastId for the next fetch
                lastId = response.last()?.id;

                // Update the progress periodically
                if (Date.now() - lastProgressUpdate > this.UPDATE_INTERVAL) {
                    await interaction.editReply(
                        UIService.formatStatusMessage(
                            EMOJIS.loading,
                            `Carregando mensagens... (${messages.length}/${limit})`
                        )
                    );
                    lastProgressUpdate = Date.now();
                }

            } catch (error: any) {
                if (error?.code === 50001) {
                    throw new Error('Não tenho permissão para ler mensagens neste canal.');
                }

                if (error?.code === 429) {
                    const retryAfter = error.retry_after * 1000 || 1000;
                    await interaction.editReply(
                        UIService.formatStatusMessage(
                            EMOJIS.network,
                            `Limite de API atingido. Aguardando ${retryAfter/1000}s...`
                        )
                    );
                    await new Promise(resolve => setTimeout(resolve, retryAfter));
                    continue;
                }

                // For non-rate-limit errors, wait a short time and retry once
                await new Promise(resolve => setTimeout(resolve, 100));
                try {
                    const retryResponse = await channel.messages.fetch({ 
                        limit: Math.min(this.MAX_BATCH_SIZE, limit - messages.length),
                        ...(lastId && { before: lastId }),
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

        // Final update
        if (messages.length > 0) {
            await interaction.editReply(
                UIService.formatStatusMessage(
                    EMOJIS.complete,
                    `Busca completa! ${messages.length} mensagens carregadas.`,
                )
            );
        }

        // Sort messages by creation time (newest first)
        messages.sort((a, b) => b.createdTimestamp - a.createdTimestamp);

        // Update cache if caching is enabled
        this.updateCache(channelId, messages);

        return messages.slice(0, limit);
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
            const cacheItems = messages.map(msg => this.messageToCache(msg));
            
            // Get existing cache or create new one
            const existing = this.loadChannelCache(channelId);
            if (existing) {
                // Create a set of existing IDs for fast lookup
                const existingIds = new Set(existing.messages.map(msg => msg.id));
                
                // Add only new messages to cache
                for (const item of cacheItems) {
                    if (!existingIds.has(item.id)) {
                        existing.messages.push(item);
                    }
                }
                
                // Update timestamp and last accessed
                existing.timestamp = Date.now();
                existing.lastAccessed = Date.now();
                this.messageCache.set(channelId, existing);
            } else {
                // Create new cache entry
                const newCache: MessageCache = {
                    messages: cacheItems,
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