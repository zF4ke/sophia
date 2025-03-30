import { ChatInputCommandInteraction, PermissionFlagsBits, GuildMember, ChannelType } from "discord.js";
import { ADMIN_IDS } from "../utils/constants";
import { FileSystemService } from "./FileSystemService";

interface RateLimitEntry {
    count: number;
    timestamp: number;
}

interface CommandConfig {
    isPublic: boolean;
    rateLimits: {
        default: number;
        admin: number;
        moderator: number;
    };
}

interface AdminUser {
    userId: string;
    addedBy: string;
    addedAt: number; 
    permissions: string[];
}

interface ModeratorUser {
    userId: string;
    addedBy: string;
    addedAt: number;
    permissions: string[];
}

/**
 * Enhanced SecurityService with admin management, command access control and rate limiting
 */
export class SecurityService {
    // Service name used for file organization
    private static readonly SERVICE_NAME = 'security';
    
    // File names
    private static readonly ADMINS_FILE = 'admins.json';
    private static readonly MODERATORS_FILE = 'moderators.json';
    private static readonly COMMANDS_CONFIG_FILE = 'commands_config.json';

    public static readonly RATE_LIMIT_WINDOW = 60000; // 1 minute in ms
    private static readonly DEFAULT_COMMAND_LIMIT = 5;
    
    private static rateLimits: Map<string, RateLimitEntry> = new Map();
    private static admins: AdminUser[] = [];
    private static moderators: ModeratorUser[] = [];
    private static commandConfigs: Map<string, CommandConfig> = new Map();
    private static isInitialized = false;

    /**
     * Initialize the SecurityService and load stored data
     */
    public static async initialize(): Promise<void> {
        if (this.isInitialized) return;
        
        try {
            // Load admins
            try {
                const adminsData = FileSystemService.readJsonFromPath<AdminUser[]>(this.ADMINS_FILE);
                
                if (adminsData) {
                    this.admins = adminsData;
                    
                    // Add hardcoded admins if they aren't in the loaded admin list
                    for (const id of ADMIN_IDS) {
                        if (!this.admins.some(admin => admin.userId === id)) {
                            this.admins.push({
                                userId: id,
                                addedBy: 'system',
                                addedAt: Date.now(),
                                permissions: ['*']
                            });
                        }
                    }
                } else {
                    // Create default admins if file doesn't exist
                    this.admins = ADMIN_IDS.map(id => ({
                        userId: id, 
                        addedBy: 'system',
                        addedAt: Date.now(),
                        permissions: ['*']
                    }));
                    await this.saveAdmins();
                }
            } catch (error) {
                console.error('Error loading admins:', error);
                // Create default admins file if it doesn't exist
                this.admins = ADMIN_IDS.map(id => ({
                    userId: id, 
                    addedBy: 'system',
                    addedAt: Date.now(),
                    permissions: ['*']
                }));
                await this.saveAdmins();
            }
            
            // Load moderators
            try {
                const moderatorsData = FileSystemService.readJsonFromPath<ModeratorUser[]>(this.MODERATORS_FILE);
                
                if (moderatorsData) {
                    this.moderators = moderatorsData;
                } else {
                    // Create empty moderators array if file doesn't exist
                    this.moderators = [];
                    await this.saveModerators();
                }
            } catch (error) {
                console.error('Error loading moderators:', error);
                // Create empty moderators file if it doesn't exist
                this.moderators = [];
                await this.saveModerators();
            }
            
            // Load command configs
            try {
                const commandConfigData = FileSystemService.readJsonFromPath<Record<string, CommandConfig>>(this.COMMANDS_CONFIG_FILE);
                
                if (commandConfigData) {
                    this.commandConfigs = new Map(Object.entries(commandConfigData));
                } else {
                    // No command configs yet, will create when needed
                    this.commandConfigs = new Map();
                    await this.saveCommandConfigs();
                }
            } catch (error) {
                console.error('Error loading command configs:', error);
                // No command configs yet, will create when needed
                this.commandConfigs = new Map();
                await this.saveCommandConfigs();
            }
            
            this.isInitialized = true;
            console.log('SecurityService inicializado com sucesso.');
        } catch (error) {
            console.error('Erro ao inicializar o SecurityService:', error);
            throw error;
        }
    }

    /**
     * Check if a user is an admin
     */
    public static isAdmin(userId: string): boolean {
        return this.admins.some(admin => admin.userId === userId);
    }
    
    /**
     * Check if a user is a moderator
     */
    public static isModerator(userId: string): boolean {
        // Admins have moderator privileges by default
        if (this.isAdmin(userId)) return true;
        
        return this.moderators.some(mod => mod.userId === userId);
    }
    
    /**
     * Add a new admin
     */
    public static async addAdmin(userId: string, addedBy: string, permissions: string[] = ['*']): Promise<boolean> {
        await this.initialize();
        
        if (this.isAdmin(userId)) return false;
        
        this.admins.push({
            userId,
            addedBy,
            addedAt: Date.now(),
            permissions
        });
        
        await this.saveAdmins();
        return true;
    }
    
    /**
     * Add a new moderator
     */
    public static async addModerator(userId: string, addedBy: string, permissions: string[] = ['moderate']): Promise<boolean> {
        await this.initialize();
        
        // If user is already an admin, they don't need to be a moderator
        if (this.isAdmin(userId)) return false;
        
        // Check if already a moderator
        if (this.isModerator(userId)) return false;
        
        this.moderators.push({
            userId,
            addedBy,
            addedAt: Date.now(),
            permissions
        });
        
        await this.saveModerators();
        return true;
    }
    
    /**
     * Remove an admin
     */
    public static async removeAdmin(userId: string): Promise<boolean> {
        await this.initialize();
        
        // If the user is not an admin, they can't be removed
        if (!SecurityService.isAdmin(userId)) return false;

        // Prevent removal of hardcoded admins
        if (ADMIN_IDS.includes(userId)) return false;

        const initialLength = this.admins.length;
        this.admins = this.admins.filter(admin => admin.userId !== userId);
        
        if (this.admins.length !== initialLength) {
            await this.saveAdmins();
            return true;
        }
        
        return false;
    }
    
    /**
     * Remove a moderator
     */
    public static async removeModerator(userId: string): Promise<boolean> {
        await this.initialize();
        
        const initialLength = this.moderators.length;
        this.moderators = this.moderators.filter(mod => mod.userId !== userId);
        
        if (this.moderators.length !== initialLength) {
            await this.saveModerators();
            return true;
        }
        
        return false;
    }

    /**
     * Set command visibility (public or private)
     */
    public static async setCommandVisibility(commandName: string, isPublic: boolean): Promise<void> {
        await this.initialize();
        
        const config = this.getCommandConfig(commandName);
        config.isPublic = isPublic;
        
        this.commandConfigs.set(commandName, config);
        await this.saveCommandConfigs();
    }
    
    /**
     * Set rate limits for a command
     */
    public static async setCommandRateLimit(
        commandName: string, 
        defaultLimit: number, 
        adminLimit: number = defaultLimit * 2,
        moderatorLimit: number = defaultLimit * 1.5
    ): Promise<void> {
        await this.initialize();
        
        const config = this.getCommandConfig(commandName);
        config.rateLimits = {
            default: defaultLimit,
            admin: adminLimit,
            moderator: moderatorLimit
        };
        
        this.commandConfigs.set(commandName, config);
        await this.saveCommandConfigs();
    }
    
    /**
     * Check if a command is public
     */
    public static async isCommandPublic(commandName: string): Promise<boolean> {
        await this.initialize();
        return this.getCommandConfig(commandName).isPublic;
    }

    /**
     * Validate channel permissions
     */
    public static async validateChannelPermissions(interaction: ChatInputCommandInteraction): Promise<boolean> {
        const channel = interaction.channel;
        if (!channel || !interaction.guild) return false;
        
        // Check if channel is a guild-based channel that supports permissions
        if (channel.type !== ChannelType.GuildText) {
            return false;
        }
        
        const botMember = interaction.guild.members.cache.get(interaction.client.user.id);
        if (!botMember) return false;

        const requiredPermissions = [
            PermissionFlagsBits.ViewChannel,
            PermissionFlagsBits.SendMessages,
            PermissionFlagsBits.ReadMessageHistory,
            PermissionFlagsBits.EmbedLinks
        ];

        return requiredPermissions.every(perm => 
            channel.permissionsFor(botMember)?.has(perm) ?? false
        );
    }

    /**
     * Check rate limit based on user role
     */
    public static async checkRateLimit(userId: string, commandName: string): Promise<boolean> {
        await this.initialize();
        
        const key = `${userId}-${commandName}`;
        const now = Date.now();
        const userLimit = this.rateLimits.get(key);
        const config = this.getCommandConfig(commandName);
        
        let limit = config.rateLimits.default;
        if (this.isAdmin(userId)) {
            limit = config.rateLimits.admin;
        } else if (this.isModerator(userId)) {
            limit = config.rateLimits.moderator;
        }

        if (!userLimit || (now - userLimit.timestamp) > this.RATE_LIMIT_WINDOW) {
            this.rateLimits.set(key, { count: 1, timestamp: now });
            return true;
        }

        if (userLimit.count >= limit) {
            return false;
        }

        userLimit.count++;
        return true;
    }
    
    /**
     * Get command usage count remaining
     */
    public static getCommandRemainingUses(userId: string, commandName: string): number {
        const key = `${userId}-${commandName}`;
        const userLimit = this.rateLimits.get(key);
        const config = this.getCommandConfig(commandName);
        
        let limit = config.rateLimits.default;
        if (this.isAdmin(userId)) {
            limit = config.rateLimits.admin;
        } else if (this.isModerator(userId)) {
            limit = config.rateLimits.moderator;
        }

        if (!userLimit || (Date.now() - userLimit.timestamp) > this.RATE_LIMIT_WINDOW) {
            return limit;
        }

        return Math.max(0, limit - userLimit.count);
    }

    /**
     * Validate member permissions
     */
    public static validateMemberPermissions(member: GuildMember, requiredPermissions: bigint[]): boolean {
        return requiredPermissions.every(perm => member.permissions.has(perm));
    }

    /**
     * Clean expired rate limits
     */
    public static cleanRateLimits(): void {
        const now = Date.now();
        for (const [key, value] of this.rateLimits.entries()) {
            if (now - value.timestamp > this.RATE_LIMIT_WINDOW) {
                this.rateLimits.delete(key);
            }
        }
    }
    
    /**
     * Get all admins
     */
    public static async getAllAdmins(): Promise<AdminUser[]> {
        await this.initialize();
        return [...this.admins];
    }
    
    /**
     * Get all moderators
     */
    public static async getAllModerators(): Promise<ModeratorUser[]> {
        await this.initialize();
        return [...this.moderators];
    }

    /**
     * Get all command configurations
     */
    public static async getCommandConfigs(): Promise<Map<string, CommandConfig>> {
        await this.initialize();
        return new Map(this.commandConfigs);
    }
    
    /**
     * Check if a command is visible (for command handler)
     * This is a synchronous version for command loading
     */
    public static isCommandVisible(commandName: string): boolean {
        try {
            // If not initialized yet, consider all commands visible
            if (!this.isInitialized) return true;
            
            // Admin commands are always visible
            if (commandName === 'access') return true;
            
            const config = this.commandConfigs.get(commandName);
            return config ? config.isPublic : true;
        } catch (error) {
            // Default to visible if there's an error
            console.error(`Error checking command visibility for ${commandName}:`, error);
            return true;
        }
    }
    
    /**
     * Check if a command's rate limit is allowed (for command handler)
     */
    public static isRateLimitAllowed(commandName: string): boolean {
        // All commands are allowed to be loaded
        return true;
    }
    
    /**
     * Save admins to file
     */
    private static async saveAdmins(): Promise<void> {
        FileSystemService.writeJsonToPath(this.ADMINS_FILE, this.admins, this.SERVICE_NAME);
    }
    
    /**
     * Save moderators to file
     */
    private static async saveModerators(): Promise<void> {
        FileSystemService.writeJsonToPath(this.MODERATORS_FILE, this.moderators, this.SERVICE_NAME);
    }
    
    /**
     * Save command configurations to file
     */
    private static async saveCommandConfigs(): Promise<void> {
        const configsObject = Object.fromEntries(this.commandConfigs);
        FileSystemService.writeJsonToPath(this.COMMANDS_CONFIG_FILE, configsObject, this.SERVICE_NAME);
    }
    
    /**
     * Get command configuration or create default
     */
    private static getCommandConfig(commandName: string): CommandConfig {
        if (!this.commandConfigs.has(commandName)) {
            this.commandConfigs.set(commandName, {
                isPublic: false,
                rateLimits: {
                    default: this.DEFAULT_COMMAND_LIMIT,
                    admin: this.DEFAULT_COMMAND_LIMIT * 2,
                    moderator: Math.floor(this.DEFAULT_COMMAND_LIMIT * 1.5)
                }
            });
        }
        
        return this.commandConfigs.get(commandName)!;
    }
}