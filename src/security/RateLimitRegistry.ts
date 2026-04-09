import type { RateLimitEntry } from "@/security/types";

export class RateLimitRegistry {
    private readonly entries = new Map<string, RateLimitEntry>();

    public checkRateLimit(key: string, limit: number, windowMs: number): boolean {
        const now = Date.now();
        const existing = this.entries.get(key);

        if (!existing || now - existing.timestamp > windowMs) {
            this.entries.set(key, { count: 1, timestamp: now });
            return true;
        }

        if (existing.count >= limit) {
            return false;
        }

        existing.count += 1;
        return true;
    }

    public getRemainingUses(key: string, limit: number, windowMs: number): number {
        const existing = this.entries.get(key);
        if (!existing || Date.now() - existing.timestamp > windowMs) {
            return limit;
        }

        return Math.max(0, limit - existing.count);
    }

    public cleanExpired(windowMs: number): void {
        const now = Date.now();
        for (const [key, value] of this.entries.entries()) {
            if (now - value.timestamp > windowMs) {
                this.entries.delete(key);
            }
        }
    }
}
