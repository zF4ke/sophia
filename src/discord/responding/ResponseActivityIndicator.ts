import type { ChatInputCommandInteraction, Message, TextBasedChannel } from "discord.js";

export type ResponseActivityIndicator = { startThinking(): Promise<void>; startTyping(): Promise<void>; stop(): Promise<void> };
const noop = (): ResponseActivityIndicator => ({ async startThinking() {}, async startTyping() {}, async stop() {} });

export class ResponseActivityService {
    static async startForInteraction(_interaction: ChatInputCommandInteraction): Promise<ResponseActivityIndicator> {
        return noop(); // Discord already displays the deferred interaction state.
    }
    static async startForMessage(_message: Message, channel: TextBasedChannel): Promise<ResponseActivityIndicator> {
        if (!("sendTyping" in channel)) return noop();
        let timer: ReturnType<typeof setInterval> | undefined;
        let generation = 0;
        let starting = false;
        const stop = async () => { generation++; if (timer) clearInterval(timer); timer = undefined; };
        const start = async () => {
            if (timer || starting) return;
            starting = true;
            const activeGeneration = generation;
            try {
                await channel.sendTyping();
                if (activeGeneration !== generation) return;
                timer = setInterval(() => { void channel.sendTyping().catch(() => {}); }, 8000);
                timer.unref?.();
            } catch { await stop(); } finally { starting = false; }
        };
        return { startThinking: start, startTyping: start, stop };
    }
}
