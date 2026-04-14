import type { DiscordToolName } from "@/shared/discordTools";
import {
    getCapability,
    listCapabilityManifests,
    describeCapabilitiesForPrompt,
    type RuntimeCapability,
} from "@/tools/registry";
import type { CapabilityManifest } from "@/runtime/contracts";

export { type RuntimeCapability };

export class CapabilityRegistry {
    public static list(): CapabilityManifest[] {
        return listCapabilityManifests();
    }

    public static get(id: DiscordToolName): RuntimeCapability {
        return getCapability(id);
    }

    public static describeForPrompt(): string {
        return describeCapabilitiesForPrompt();
    }
}

