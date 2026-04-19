import {
    ChannelType,
    ChatInputCommandInteraction,
    GuildMember,
    PermissionFlagsBits,
} from "discord.js";
import { AdminDirectory } from "@/security/AdminDirectory";
import { CommandPolicyRegistry } from "@/security/CommandPolicyRegistry";
import { ACCESS_POLICY_TARGETS, type AccessPolicyTarget } from "@/security/policyTargets";
import { RateLimitRegistry } from "@/security/RateLimitRegistry";
import type { AdminUser, CommandConfig, ModeratorUser } from "@/security/types";

export class SecurityService {
    public static readonly RATE_LIMIT_WINDOW = 15000;

    private static readonly adminDirectory = new AdminDirectory();
    private static readonly commandPolicyRegistry = new CommandPolicyRegistry();
    private static readonly rateLimitRegistry = new RateLimitRegistry();
    private static isInitialized = false;

    public static async initialize(): Promise<void> {
        if (this.isInitialized) {
            return;
        }

        this.adminDirectory.load();
        this.commandPolicyRegistry.load();
        this.isInitialized = true;
    }

    public static isAdmin(userId: string): boolean {
        return this.adminDirectory.isAdmin(userId);
    }

    public static isModerator(userId: string): boolean {
        return this.adminDirectory.isModerator(userId);
    }

    public static async addAdmin(
        userId: string,
        addedBy: string,
        permissions: string[] = ["*"]
    ): Promise<boolean> {
        await this.initialize();
        return this.adminDirectory.addAdmin(userId, addedBy, permissions);
    }

    public static async removeAdmin(userId: string): Promise<boolean> {
        await this.initialize();
        return this.adminDirectory.removeAdmin(userId);
    }

    public static async addModerator(
        userId: string,
        addedBy: string,
        permissions: string[] = ["moderate"]
    ): Promise<boolean> {
        await this.initialize();
        return this.adminDirectory.addModerator(userId, addedBy, permissions);
    }

    public static async removeModerator(userId: string): Promise<boolean> {
        await this.initialize();
        return this.adminDirectory.removeModerator(userId);
    }

    public static async setCommandVisibility(
        commandName: string,
        isPublic: boolean
    ): Promise<void> {
        await this.initialize();
        this.commandPolicyRegistry.setCommandVisibility(commandName, isPublic);
    }

    public static async setCommandRateLimit(
        commandName: string,
        defaultLimit: number,
        adminLimit = defaultLimit * 2,
        moderatorLimit = defaultLimit * 1.5
    ): Promise<void> {
        await this.initialize();
        this.commandPolicyRegistry.setCommandRateLimit(
            commandName,
            defaultLimit,
            adminLimit,
            moderatorLimit
        );
    }

    public static async isCommandPublic(commandName: string): Promise<boolean> {
        await this.initialize();
        return this.commandPolicyRegistry.isCommandPublic(commandName);
    }

    public static async isTriggerEnabled(
        trigger: AccessPolicyTarget,
        userId?: string,
    ): Promise<boolean> {
        await this.initialize();
        if (userId && this.isAdmin(userId)) {
            return true;
        }
        return this.commandPolicyRegistry.isCommandPublic(trigger);
    }

    public static async checkRateLimit(
        userId: string,
        commandName: string
    ): Promise<boolean> {
        await this.initialize();
        if (this.isAdmin(userId)) {
            return true;
        }
        const config = this.commandPolicyRegistry.getCommandConfig(commandName);

        let limit = config.rateLimits.default;
        if (this.isAdmin(userId)) {
            limit = config.rateLimits.admin;
        } else if (this.isModerator(userId)) {
            limit = config.rateLimits.moderator;
        }

        return this.rateLimitRegistry.checkRateLimit(
            `${userId}-${commandName}`,
            limit,
            this.RATE_LIMIT_WINDOW
        );
    }

    public static async checkTriggerRateLimit(
        userId: string,
        trigger: AccessPolicyTarget,
    ): Promise<boolean> {
        return this.checkRateLimit(userId, trigger);
    }

    public static getCommandRemainingUses(userId: string, commandName: string): number {
        if (this.isAdmin(userId)) {
            return Number.POSITIVE_INFINITY;
        }
        const config = this.commandPolicyRegistry.getCommandConfig(commandName);

        let limit = config.rateLimits.default;
        if (this.isAdmin(userId)) {
            limit = config.rateLimits.admin;
        } else if (this.isModerator(userId)) {
            limit = config.rateLimits.moderator;
        }

        return this.rateLimitRegistry.getRemainingUses(
            `${userId}-${commandName}`,
            limit,
            this.RATE_LIMIT_WINDOW
        );
    }

    public static cleanRateLimits(): void {
        this.rateLimitRegistry.cleanExpired(this.RATE_LIMIT_WINDOW);
    }

    public static async getAllAdmins(): Promise<AdminUser[]> {
        await this.initialize();
        return this.adminDirectory.getAllAdmins();
    }

    public static async getAllModerators(): Promise<ModeratorUser[]> {
        await this.initialize();
        return this.adminDirectory.getAllModerators();
    }

    public static async getCommandConfigs(): Promise<Map<string, CommandConfig>> {
        await this.initialize();
        return this.commandPolicyRegistry.getCommandConfigs();
    }

    public static getPolicyTargetLabels(): Record<AccessPolicyTarget, string> {
        return {
            [ACCESS_POLICY_TARGETS.mention]: "@mention",
            [ACCESS_POLICY_TARGETS.reply]: "reply",
        };
    }

    public static async validateChannelPermissions(
        interaction: ChatInputCommandInteraction
    ): Promise<boolean> {
        const channel = interaction.channel;
        if (!channel || !interaction.guild || channel.type !== ChannelType.GuildText) {
            return false;
        }

        const botMember = interaction.guild.members.cache.get(interaction.client.user.id);
        if (!botMember) {
            return false;
        }

        const requiredPermissions = [
            PermissionFlagsBits.ViewChannel,
            PermissionFlagsBits.SendMessages,
            PermissionFlagsBits.ReadMessageHistory,
            PermissionFlagsBits.EmbedLinks,
        ];

        return requiredPermissions.every(
            (permission) => channel.permissionsFor(botMember)?.has(permission) ?? false
        );
    }

    public static validateMemberPermissions(
        member: GuildMember,
        requiredPermissions: bigint[]
    ): boolean {
        return requiredPermissions.every((permission) => member.permissions.has(permission));
    }
}
