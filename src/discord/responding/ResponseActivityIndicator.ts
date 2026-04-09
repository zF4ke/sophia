import type {
    ChatInputCommandInteraction,
    TextBasedChannel,
} from "discord.js";

const TYPING_INTERVAL_MS = 8_000;

export type ResponseActivityIndicator = {
    stop(): void;
};

function createNoopIndicator(): ResponseActivityIndicator {
    return {
        stop() {
            return;
        },
    };
}

export class ResponseActivityService {
    public static async startForInteraction(
        _interaction: ChatInputCommandInteraction
    ): Promise<ResponseActivityIndicator> {
        return createNoopIndicator();
    }

    public static async startForChannel(
        channel: TextBasedChannel
    ): Promise<ResponseActivityIndicator> {
        if (!("sendTyping" in channel) || typeof channel.sendTyping !== "function") {
            return createNoopIndicator();
        }

        try {
            await channel.sendTyping();
        } catch {
            return createNoopIndicator();
        }

        const timer = setInterval(() => {
            void channel.sendTyping().catch(() => undefined);
        }, TYPING_INTERVAL_MS);

        return {
            stop() {
                clearInterval(timer);
            },
        };
    }
}
