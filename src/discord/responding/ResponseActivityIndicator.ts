import type {
    ChatInputCommandInteraction,
    Message,
    TextBasedChannel,
} from "discord.js";

const TYPING_INTERVAL_MS = 8_000;
const THINKING_EMOJI_FRAMES = [
    "🤨",
    "🧐",
    "🤓",
    "😎",
    "🤔",
    "🫡",
    "😴",
    "😬",
] as const;
const THINKING_ANIMATION_INTERVAL_MS = 2_000;

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
        let thinkingTimer: ReturnType<typeof setInterval> | null = null;
        let thinkingFrameIndex = 0;
        let currentThinkingEmoji: string | null = null;

        const stopTyping = () => {
            if (typingTimer) {
                clearInterval(typingTimer);
                typingTimer = null;
            }
        };

        const stopThinking = () => {
            if (thinkingTimer) {
                clearInterval(thinkingTimer);
                thinkingTimer = null;
            }
        };

        const removeThinkingReaction = async () => {
            stopThinking();
            if (!currentThinkingEmoji) {
                return;
            }
            try {
                const botUserId = message.client.user?.id;
                const reaction = message.reactions.resolve(currentThinkingEmoji);
                if (botUserId && reaction) {
                    await reaction.users.remove(botUserId);
                }
            } catch {
                // Ignore cleanup failures due to missing permissions/race.
            } finally {
                currentThinkingEmoji = null;
            }
        };

        const tickThinkingFrame = async () => {
            const nextEmoji = THINKING_EMOJI_FRAMES[thinkingFrameIndex];
            thinkingFrameIndex =
                (thinkingFrameIndex + 1) % THINKING_EMOJI_FRAMES.length;

            try {
                const botUserId = message.client.user?.id;
                const previousEmoji = currentThinkingEmoji;
                await message.react(nextEmoji);
                currentThinkingEmoji = nextEmoji;
                if (previousEmoji && botUserId && previousEmoji !== nextEmoji) {
                    const prevReaction = message.reactions.resolve(
                        previousEmoji,
                    );
                    if (prevReaction) {
                        await prevReaction.users.remove(botUserId);
                    }
                }
            } catch {
                // Missing reaction permissions is acceptable.
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
            await removeThinkingReaction();
            thinkingFrameIndex = 0;
            await tickThinkingFrame();
            thinkingTimer = setInterval(() => {
                void tickThinkingFrame();
            }, THINKING_ANIMATION_INTERVAL_MS);
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
