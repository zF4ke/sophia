import type { AccessPanelView } from "./panelTypes";

export const ACCESS_PANEL_PREFIX = "access";

export const ACCESS_VIEW_BUTTONS = {
    overview: `${ACCESS_PANEL_PREFIX}:view:overview`,
    admins: `${ACCESS_PANEL_PREFIX}:view:admins`,
    moderators: `${ACCESS_PANEL_PREFIX}:view:moderators`,
    commands: `${ACCESS_PANEL_PREFIX}:view:commands`,
} as const satisfies Record<AccessPanelView, string>;

export const ACCESS_ADMIN_ADD_ID = `${ACCESS_PANEL_PREFIX}:admins:add`;
export const ACCESS_ADMIN_REMOVE_ID = `${ACCESS_PANEL_PREFIX}:admins:remove`;
export const ACCESS_MODERATOR_ADD_ID = `${ACCESS_PANEL_PREFIX}:moderators:add`;
export const ACCESS_MODERATOR_REMOVE_ID = `${ACCESS_PANEL_PREFIX}:moderators:remove`;
export const ACCESS_COMMAND_SELECT_ID = `${ACCESS_PANEL_PREFIX}:commands:select`;

export function getAccessVisibilityId(commandName: string, isPublic: boolean): string {
    return `${ACCESS_PANEL_PREFIX}:commands:visibility:${commandName}:${isPublic ? "public" : "private"}`;
}

export function getAccessLimitsButtonId(commandName: string): string {
    return `${ACCESS_PANEL_PREFIX}:commands:limits:${commandName}`;
}

export function getAccessLimitsModalId(commandName: string): string {
    return `${ACCESS_PANEL_PREFIX}:commands:limits_modal:${commandName}`;
}

export function isAccessInteraction(customId: string): boolean {
    return customId.startsWith(`${ACCESS_PANEL_PREFIX}:`);
}
