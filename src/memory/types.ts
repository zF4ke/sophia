export interface StoredMessage {
    id: string;
    guildId: string | null;
    channelId: string;
    channelName: string;
    authorId: string;
    authorName: string;
    content: string;
    attachmentsJson: string;
    referenceMessageId: string | null;
    createdTimestamp: number;
    jumpLink: string;
    isBot: number;
}

export interface KnownChannelRecord {
    channelId: string;
    guildId: string | null;
    channelName: string;
    channelType: string | null;
    parentCategoryId: string | null;
    parentCategoryName: string | null;
    lastSeenTimestamp: number;
}

export interface HistoricalAuthorRecord {
    authorId: string;
    authorName: string;
    guildId: string | null;
    messageCount: number;
    lastSeenTimestamp: number;
    isBot: boolean;
}

export interface SearchMessageScope {
    guildId?: string | null;
    channelIds?: string[];
    authorIds?: string[];
    beforeTimestamp?: number | null;
    afterTimestamp?: number | null;
    excludedMessageIds?: string[];
}

export interface ChannelIndexState {
    channelId: string;
    lastMessageId: string | null;
    lastIndexedTimestamp: number | null;
}

export interface ChannelCrawlState {
    channelId: string;
    lastCrawledTimestamp: number | null;
    oldestFetchedMessageId: string | null;
    exhausted: boolean;
}
