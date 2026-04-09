import { SecurityConfigStore } from "@/security/storage/SecurityConfigStore";
import type { CommandConfig } from "@/security/types";

export class CommandPolicyRegistry {
    private static readonly DEFAULT_COMMAND_LIMIT = 5;
    private commandConfigs = new Map<string, CommandConfig>();

    public load(): void {
        this.commandConfigs = SecurityConfigStore.loadCommandConfigs();
    }

    public setCommandVisibility(commandName: string, isPublic: boolean): void {
        const config = this.getCommandConfig(commandName);
        config.isPublic = isPublic;
        this.commandConfigs.set(commandName, config);
        SecurityConfigStore.saveCommandConfigs(this.commandConfigs);
    }

    public setCommandRateLimit(
        commandName: string,
        defaultLimit: number,
        adminLimit: number,
        moderatorLimit: number
    ): void {
        const config = this.getCommandConfig(commandName);
        config.rateLimits = {
            default: defaultLimit,
            admin: adminLimit,
            moderator: moderatorLimit,
        };
        this.commandConfigs.set(commandName, config);
        SecurityConfigStore.saveCommandConfigs(this.commandConfigs);
    }

    public isCommandPublic(commandName: string): boolean {
        return this.getCommandConfig(commandName).isPublic;
    }

    public getCommandConfigs(): Map<string, CommandConfig> {
        return new Map(this.commandConfigs);
    }

    public getCommandConfig(commandName: string): CommandConfig {
        if (!this.commandConfigs.has(commandName)) {
            this.commandConfigs.set(commandName, {
                isPublic: false,
                rateLimits: {
                    default: CommandPolicyRegistry.DEFAULT_COMMAND_LIMIT,
                    admin: CommandPolicyRegistry.DEFAULT_COMMAND_LIMIT * 2,
                    moderator: Math.floor(CommandPolicyRegistry.DEFAULT_COMMAND_LIMIT * 1.5),
                },
            });
        }

        return this.commandConfigs.get(commandName)!;
    }
}
