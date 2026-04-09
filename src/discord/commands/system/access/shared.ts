import {
    ChatInputCommandInteraction,
    MessageFlags,
    type InteractionReplyOptions,
    type User,
} from "discord.js";

export async function replyWithAccessMessage(
    interaction: ChatInputCommandInteraction,
    content: string,
    ephemeral: boolean,
    options: Omit<InteractionReplyOptions, "content" | "flags"> = {}
): Promise<void> {
    await interaction.reply({
        content,
        flags: ephemeral ? MessageFlags.Ephemeral : undefined,
        ...options,
    });
}

export async function requireTargetUser(
    interaction: ChatInputCommandInteraction,
    ephemeral: boolean
): Promise<User | null> {
    const user = interaction.options.getUser("user");
    if (user) {
        return user;
    }

    await replyWithAccessMessage(interaction, "❌ Usuário não encontrado", ephemeral);
    return null;
}

export async function fetchUserUsername(
    interaction: ChatInputCommandInteraction,
    userId: string
): Promise<string> {
    const cached = interaction.client.users.cache.get(userId);
    if (cached) {
        return cached.username;
    }

    const fetched = await interaction.client.users.fetch(userId).catch(() => null);
    return fetched?.username || userId;
}
