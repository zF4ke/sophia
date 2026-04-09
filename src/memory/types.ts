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

export interface ChannelIndexState {
    channelId: string;
    lastMessageId: string | null;
    lastIndexedTimestamp: number | null;
}

export interface SearchMessageScope {
    guildId?: string | null;
    channelIds?: string[];
    authorIds?: string[];
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
