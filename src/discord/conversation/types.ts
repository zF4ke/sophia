import { Message } from "discord.js";

export interface ConversationWithContext {
    messages: Message[];
    relevanceScore: number;
}

export interface MessageGroup {
    author: string;
    content: string[];
    timestamp: number;
}

export interface AIAnalysisResult {
    isRelevant: boolean;
    relevanceScore: number;
}

export interface DisplayableItem {
    getFieldName(): string;
    getFieldContent(): string[];
    getMetadata?(): Record<string, string | number>;
    getLink?(): string | undefined;
}

export interface PaginationOptions {
    itemsPerPage?: number;
    collectorTimeout?: number;
    ephemeral?: boolean;
    color?: number;
}