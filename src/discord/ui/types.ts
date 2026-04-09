import type { Message } from "discord.js";

export interface NavigationButton {
    customId: string;
    label: string;
    style: "Primary" | "Secondary" | "Success" | "Danger";
    disabled: boolean;
}

export interface NavigationState {
    currentConvIndex: number;
    currentMsgIndex: number;
}

export interface MessageGroup {
    author: string;
    content: string[];
    timestamp: number;
}

export interface LongMessageTarget {
    editReply(payload: { content: string }): Promise<unknown>;
    followUp(payload: { content: string; flags?: number }): Promise<unknown>;
}

export interface ReplyMessageTarget {
    reply(payload: {
        content: string;
        allowedMentions?: {
            parse: string[];
            repliedUser: boolean;
        };
    }): Promise<Message>;
}
