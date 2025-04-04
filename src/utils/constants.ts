export const EMOJIS = {
    // Original emojis
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
    info: "📖",
    delete: "🗑️",
    cache: "📦",
    loading: "⏳",
    merge: "🔄",
    found: "🔎",
    database: "🗄️",
    network: "📡",
    complete: "✨",
    memory: "💾",
    check: "✅",
    sync: "🔄",
    filter: "🔧",
    settings: "⚙️",
    help: "❓",
    edit: "✏️",
    add: "➕",
    remove: "➖",
    share: "🔗"
} as const;

// Remove the 'as const' to allow string comparison
export const ADMIN_IDS = [
    "676156690395037713",
    "111591984245780480"
];

/**
 * Discord platform specific constants
 */
export const DISCORD = {
    /**
     * Maximum character limit for a single Discord message
     * Messages exceeding this limit need to be split
     */
    MESSAGE_LIMIT: 2000
};

export default { EMOJIS, ADMIN_IDS, DISCORD };