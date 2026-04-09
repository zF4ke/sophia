import { getAppConfig } from "@/app/AppConfig";
import type { Message } from "discord.js";
import { ModelGateway } from "@/ai/ModelGateway";
import { MessageEligibility } from "@/memory/ingest/MessageEligibility";
import { MessageNormalizer } from "@/memory/ingest/MessageNormalizer";
import { MessageChunker } from "@/memory/index/MessageChunker";
import { CacheReadRepository } from "@/memory/repositories/CacheReadRepository";
import { CacheWriteRepository } from "@/memory/repositories/CacheWriteRepository";
import { MemoryReadRepository } from "@/memory/repositories/MemoryReadRepository";
import { MemoryWriteRepository } from "@/memory/repositories/MemoryWriteRepository";
import { ResponseCounterRepository } from "@/memory/repositories/ResponseCounterRepository";
import { MemorySearchService } from "@/memory/search/MemorySearchService";
import { SearchScopeClause } from "@/memory/search/SearchScopeClause";
import type {
    CachedToolResultRecord,
    ChannelCrawlState,
    ConversationResolutionContextRecord,
    ReusableGroundedContextRecord,
    SearchMessageScope,
    StoredMessage,
} from "@/memory/types";
import type { ChannelCandidate, RetrievedChunk } from "@/shared/appTypes";

const EMBEDDING_BATCH_SIZE = 32;

export class DiscordMemoryService {
    public static isEligibleMessage(message: Message): boolean {
        return MessageEligibility.isEligible(message);
    }

    public static async ingestMessage(message: Message): Promise<void> {
        if (!this.isEligibleMessage(message)) {
            return;
        }

        await this.ingestStoredMessage(MessageNormalizer.toStoredMessage(message));
    }

    public static async ingestStoredMessage(stored: StoredMessage): Promise<void> {
        const chunks = MessageChunker.split(stored.content);
        MemoryWriteRepository.upsertMessageWithChunks(stored, chunks);

        const model = getAppConfig().modelProfile.embeddingModel;
        for (let index = 0; index < chunks.length; index += EMBEDDING_BATCH_SIZE) {
            const batch = chunks.slice(index, index + EMBEDDING_BATCH_SIZE);
            const vectors = await ModelGateway.embedTexts(batch);
            MemoryWriteRepository.upsertEmbeddings(stored.id, index, model, vectors);
        }

        MemoryWriteRepository.updateIndexState(
            stored.channelId,
            stored.id,
            stored.createdTimestamp
        );
    }

    public static getIndexState(channelId?: string) {
        return MemoryReadRepository.getIndexState(channelId);
    }

    public static getChannelCrawlState(channelId?: string): ChannelCrawlState[] {
        return MemoryReadRepository.getChannelCrawlState(channelId);
    }

    public static clearAll(): void {
        MemoryWriteRepository.clearAll();
    }

    public static getStats() {
        return MemoryReadRepository.getStats();
    }

    public static async searchMessagesAsync(
        query: string,
        scope: SearchMessageScope = {},
        limit = 8
    ): Promise<RetrievedChunk[]> {
        return MemorySearchService.searchMessages(query, scope, limit);
    }

    public static getMessageThread(messageId: string, window = 6) {
        return MemoryReadRepository.getMessageThread(messageId, window);
    }

    public static getChannelSummary(channelId: string) {
        return MemoryReadRepository.getChannelSummary(channelId);
    }

    public static listRelevantChannels(
        query: string,
        scope: SearchMessageScope = {},
        limit = 6
    ) {
        const ftsQuery = SearchScopeClause.toFtsQuery(query);
        if (!ftsQuery) {
            return [];
        }

        const scopeClause = SearchScopeClause.build(scope);
        return MemoryReadRepository.listRelevantChannels(
            ftsQuery,
            scopeClause.sql,
            scopeClause.params,
            limit
        );
    }

    public static getKnownChannels(guildId?: string | null) {
        return MemoryReadRepository.listKnownChannels(guildId);
    }

    public static upsertDiscoveredChannel(
        channelId: string,
        guildId: string | null,
        channelName: string,
        timestamp = Date.now()
    ): void {
        MemoryWriteRepository.upsertDiscoveredChannel(
            channelId,
            guildId,
            channelName,
            timestamp
        );
    }

    public static updateChannelCrawlState(
        channelId: string,
        oldestFetchedMessageId: string | null,
        exhausted: boolean
    ): void {
        MemoryWriteRepository.updateChannelCrawlState(
            channelId,
            oldestFetchedMessageId,
            exhausted
        );
    }

    public static getNthHistoricalMessage(channelId: string, position: number) {
        return MemoryReadRepository.getNthHistoricalMessage(channelId, position);
    }

    public static recordToolRun(
        guildId: string | null,
        channelId: string | null,
        userId: string,
        question: string,
        toolName: string,
        summary: string
    ): void {
        MemoryWriteRepository.recordToolRun(
            guildId,
            channelId,
            userId,
            question,
            toolName,
            summary
        );
    }

    public static repairIndexes(): void {
        MemoryWriteRepository.repairIndexes();
    }

    public static getReusableGroundedContext(options: {
        guildId: string | null;
        currentChannelId: string | null;
        questionFingerprint: string;
        routeIntent: string;
        requireSufficient?: boolean;
        currentResponseOrdinal?: number | null;
        maxResponsesAgo?: number;
    }): ReusableGroundedContextRecord | null {
        return CacheReadRepository.getReusableGroundedContext(options);
    }

    public static saveReusableGroundedContext(
        context: ReusableGroundedContextRecord
    ): void {
        CacheWriteRepository.upsertReusableGroundedContext(context);
    }

    public static getCachedToolResult(
        cacheKey: string,
        currentResponseOrdinal: number | null,
        maxResponsesAgo: number
    ): CachedToolResultRecord | null {
        return CacheReadRepository.getCachedToolResult(
            cacheKey,
            currentResponseOrdinal,
            maxResponsesAgo
        );
    }

    public static saveCachedToolResult(entry: CachedToolResultRecord): void {
        CacheWriteRepository.upsertCachedToolResult(entry);
    }

    public static pruneCacheEntries(now = Date.now()): void {
        CacheWriteRepository.pruneExpired(now);
    }

    public static nextGuildResponseOrdinal(guildId: string | null): number | null {
        return ResponseCounterRepository.nextGuildResponseOrdinal(guildId);
    }

    public static getConversationResolutionContext(options: {
        guildId: string | null;
        currentChannelId: string | null;
        currentResponseOrdinal?: number | null;
        maxResponsesAgo?: number;
    }): ConversationResolutionContextRecord | null {
        return CacheReadRepository.getConversationResolutionContext(options);
    }

    public static saveConversationResolutionContext(
        context: ConversationResolutionContextRecord
    ): void {
        CacheWriteRepository.upsertConversationResolutionContext(context);
    }
}
