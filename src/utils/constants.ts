export const EMOJIS = {
    search: "🔍",
    conversation: "💬",
    page: "📄",
    relevance: "⭐",
    channel: "📌",
    time: "⏱️",
    error: "❌",
    success: "✅",
    warning: "⚠️",
    wave: "👋",
    info: "ℹ️"
} as const;

// Remove the 'as const' to allow string comparison
export const ADMIN_IDS = [
    "676156690395037713",
    "111591984245780480"
];

export default { EMOJIS, ADMIN_IDS };