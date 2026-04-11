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
