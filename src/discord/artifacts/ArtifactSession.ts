import { randomUUID } from "crypto";
import { MessageFlags, type Message, type SendableChannels } from "discord.js";
import {
    artifactExpiryTimestamp,
    attachmentFileNameFromUrl,
    buildArtifactComponents,
    type ArtifactSpec,
} from "./ArtifactBuilder";
import { ArtifactStore } from "./ArtifactStore";

export interface SentArtifact {
    messageId: string;
    channelId: string;
    messageUrl: string;
    persistenceError?: string;
}

/**
 * Sends an artifact card. Navigation state persists in the artifacts store
 * and interactions are handled centrally (see ArtifactInteractions), so
 * buttons keep working across restarts for the life of the message.
 */
export async function sendInteractiveArtifact(
    channel: SendableChannels,
    spec: ArtifactSpec,
    opts: { guildId: string | null; ttlDays: number; ownerId?: string | null; sources?: string[] },
): Promise<SentArtifact> {
    const nonce = newNonce();
    const built = buildArtifactComponents(spec, { section: 0 }, nonce);
    // File components reference attachment://<name>, so the actual files must
    // be uploaded with the message and named deterministically.
    const files = spec.files?.length
        ? spec.files.map((url, index) => ({ attachment: url, name: attachmentFileNameFromUrl(url, index) }))
        : undefined;
    const message: Message = await channel.send({
        components: built.components,
        ...(files ? { files } : {}),
        flags: MessageFlags.IsComponentsV2,
    } as never);

    const expiresAt = artifactExpiryTimestamp(opts.ttlDays);
    try { await ArtifactStore.record({
            messageId: message.id,
            channelId: channel.id,
              guildId: opts.guildId,
              ownerId: opts.ownerId,
            sources: opts.sources,
            expiresAt,
            specJson: JSON.stringify(spec),
        }); } catch {
        return { messageId: message.id, channelId: channel.id, messageUrl: message.url,
            persistenceError: `Card message ${message.id} was sent, but its editable state could not be saved. Do not send a duplicate; inspect the existing message.` };
    }

    return { messageId: message.id, channelId: channel.id, messageUrl: message.url };
}

/**
 * Re-render an existing card in place with a new spec. View state resets to
 * the first section; interaction handling is unchanged (central, DB-backed).
 */
export async function applyArtifactEdit(message: Message, spec: ArtifactSpec): Promise<SentArtifact> {
    const nonce = newNonce();
    const built = buildArtifactComponents(spec, { section: 0 }, nonce);
    await message.edit({
        components: built.components,
        flags: MessageFlags.IsComponentsV2,
    });
    return { messageId: message.id, channelId: message.channelId, messageUrl: message.url };
}

function newNonce(): string {
    return randomUUID().replace(/-/g, "").slice(0, 12);
}
