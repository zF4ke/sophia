export type AccessPanelView = "overview" | "admins" | "moderators" | "commands";

export interface AccessPanelState {
    view: AccessPanelView;
    selectedCommand?: string;
    notice?: string;
}
