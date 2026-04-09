import { getAppConfig } from "@/app/AppConfig";
import type { Message } from "discord.js";
import { ModelGateway } from "@/ai/ModelGateway";
import { MessageEligibility } from "@/memory/ingest/MessageEligibility";
import { MessageNormalizer } from "@/memory/ingest/MessageNormalizer";
import { MessageChunker } from "@/memory/index/MessageChunker";
import { MemoryReadRepository } from "@/memory/repositories/MemoryReadRepository";
import { MemoryWriteRepository } from "@/memory/repositories/MemoryWriteRepository";
import { MemorySearchService } from "@/memory/search/MemorySearchService";
import { SearchScopeClause } from "@/memory/search/SearchScopeClause";
import type { SearchMessageScope, StoredMessage } from "@/memory/types";
import type { RetrievedChunk } from "@/shared/appTypes";

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
        const scopeClause = SearchScopeClause.build(scope);
        return MemoryReadRepository.listRelevantChannels(
            SearchScopeClause.toFtsQuery(query),
            scopeClause.sql,
            scopeClause.params,
            limit
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
}
