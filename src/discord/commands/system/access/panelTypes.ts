export type AccessPanelView = "overview" | "admins" | "moderators" | "commands" | "grants";

export interface AccessPanelState {
    view: AccessPanelView;
    selectedCommand?: string;
    notice?: string;
    page?: number;
}
