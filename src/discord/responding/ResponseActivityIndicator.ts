import type {
    ChatInputCommandInteraction,
    Message,
    TextBasedChannel,
} from "discord.js";

const TYPING_INTERVAL_MS = 8_000;
const THINKING_EMOJI = "🤔";

export type ResponseActivityIndicator = {
    startThinking(): Promise<void>;
    startTyping(): Promise<void>;
    stop(): Promise<void>;
};

function createNoopIndicator(): ResponseActivityIndicator {
    return {
        async startThinking() {
            return;
        },
        async startTyping() {
            return;
        },
        async stop() {
            return;
        },
    };
}

export class ResponseActivityService {
    public static async startForInteraction(
        _interaction: ChatInputCommandInteraction
    ): Promise<ResponseActivityIndicator> {
        // Discord already shows a native "thinking" state after deferReply.
        return createNoopIndicator();
    }

    public static async startForMessage(
        message: Message,
        channel: TextBasedChannel,
    ): Promise<ResponseActivityIndicator> {
        if (!("sendTyping" in channel) || typeof channel.sendTyping !== "function") {
            return createNoopIndicator();
        }

        let typingTimer: ReturnType<typeof setInterval> | null = null;
        let thinkingReactionAdded = false;

        const stopTyping = () => {
            if (typingTimer) {
                clearInterval(typingTimer);
                typingTimer = null;
            }
        };

        const removeThinkingReaction = async () => {
            if (!thinkingReactionAdded) {
                return;
            }
            thinkingReactionAdded = false;
            try {
                const botUserId = message.client.user?.id;
                const reaction = message.reactions.resolve(THINKING_EMOJI);
                if (botUserId && reaction) {
                    await reaction.users.remove(botUserId);
                }
            } catch {
                // Ignore cleanup failures due to missing permissions/race.
            }
        };

        const startTyping = async () => {
            stopTyping();
            try {
                await channel.sendTyping();
                typingTimer = setInterval(() => {
                    void channel.sendTyping().catch(() => undefined);
                }, TYPING_INTERVAL_MS);
            } catch {
                stopTyping();
            }
        };

        const startThinking = async () => {
            stopTyping();
            try {
                await message.react(THINKING_EMOJI);
                thinkingReactionAdded = true;
            } catch {
                // Missing Add Reactions permission is acceptable.
            }
        };

        return {
            startThinking,
            async startTyping() {
                await removeThinkingReaction();
                await startTyping();
            },
            async stop() {
                stopTyping();
                await removeThinkingReaction();
            },
        };
    }
}
