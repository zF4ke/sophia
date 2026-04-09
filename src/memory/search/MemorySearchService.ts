import { MemoryDatabase } from "@/memory/MemoryDatabase";
import { ModelGateway } from "@/ai/ModelGateway";
import { MemoryReadRepository } from "@/memory/repositories/MemoryReadRepository";
import { SearchScopeClause } from "@/memory/search/SearchScopeClause";
import type { RetrievedChunk } from "@/shared/appTypes";
import type { SearchMessageScope } from "@/memory/types";

export class MemorySearchService {
    public static async searchMessages(
        query: string,
        scope: SearchMessageScope = {},
        limit = 8
    ): Promise<RetrievedChunk[]> {
        const normalizedQuery = query.trim();
        if (!normalizedQuery) {
            return [];
        }

        const scopeClause = SearchScopeClause.build(scope);
        const lexicalRows = MemoryDatabase.get()
            .prepare(`
                SELECT
                    mc.chunk_id,
                    mc.message_id,
                    mc.channel_id,
                    m.channel_name,
                    mc.guild_id,
                    m.author_id,
                    m.author_name,
                    mc.content,
                    mc.created_timestamp,
                    m.jump_link,
                    bm25(chunk_fts) AS lexical_rank
                FROM chunk_fts
                JOIN message_chunks mc ON chunk_fts.chunk_id = mc.chunk_id
                JOIN messages m ON m.id = mc.message_id
                WHERE chunk_fts MATCH ?
                ${scopeClause.sql}
                ORDER BY lexical_rank
                LIMIT ?
            `)
            .all(
                SearchScopeClause.toFtsQuery(normalizedQuery),
                ...scopeClause.params,
                Math.max(limit * 5, 20)
            ) as Array<Record<string, unknown>>;

        if (!lexicalRows.length) {
            return [];
        }

        const [queryVector] = await ModelGateway.embedTexts([normalizedQuery]);
        return lexicalRows
            .map((row) => this.rankRow(row, queryVector))
            .sort((a, b) => b.totalScore - a.totalScore)
            .slice(0, limit);
    }

    private static rankRow(row: Record<string, unknown>, queryVector: number[]): RetrievedChunk {
        const embedding = MemoryReadRepository.getEmbedding(String(row.chunk_id));
        const semanticScore = embedding.length ? this.cosineSimilarity(queryVector, embedding) : 0;
        const lexicalScore = Math.max(0.05, 1 / (1 + Math.abs(Number(row.lexical_rank || 0))));
        const ageMs = Date.now() - Number(row.created_timestamp);
        const recencyScore = 1 / (1 + ageMs / (1000 * 60 * 60 * 24 * 30));
        const totalScore = lexicalScore * 0.45 + semanticScore * 0.45 + recencyScore * 0.1;

        return {
            messageId: String(row.message_id),
            channelId: String(row.channel_id),
            channelName: String(row.channel_name),
            guildId: row.guild_id ? String(row.guild_id) : null,
            authorId: String(row.author_id),
            authorName: String(row.author_name),
            content: String(row.content),
            createdTimestamp: Number(row.created_timestamp),
            jumpLink: String(row.jump_link),
            lexicalScore,
            semanticScore,
            recencyScore,
            totalScore,
        };
    }

    private static cosineSimilarity(a: number[], b: number[]): number {
        if (!a.length || !b.length || a.length !== b.length) {
            return 0;
        }

        let dot = 0;
        let magnitudeA = 0;
        let magnitudeB = 0;

        for (let index = 0; index < a.length; index += 1) {
            dot += a[index] * b[index];
            magnitudeA += a[index] * a[index];
            magnitudeB += b[index] * b[index];
        }

        if (!magnitudeA || !magnitudeB) {
            return 0;
        }

        return dot / (Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB));
    }
}
