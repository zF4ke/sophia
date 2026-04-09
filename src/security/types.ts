export interface RateLimitEntry {
    count: number;
    timestamp: number;
}

export interface CommandConfig {
    isPublic: boolean;
    rateLimits: {
        default: number;
        admin: number;
        moderator: number;
    };
}

export interface AdminUser {
    userId: string;
    addedBy: string;
    addedAt: number;
    permissions: string[];
}

export interface ModeratorUser {
    userId: string;
    addedBy: string;
    addedAt: number;
    permissions: string[];
}
