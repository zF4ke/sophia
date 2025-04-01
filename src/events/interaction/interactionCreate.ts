import { TDiscordClient } from "@/index";
import { Interaction, MessageFlags } from "discord.js";
import { SecurityService } from "../../services/SecurityService";
import { readdirSync } from "fs";
import { join } from "path";

module.exports = {
    name: "interactionCreate",
    async execute(interaction: Interaction, client: TDiscordClient) {
        // Handle autocomplete interactions
        if (interaction.isAutocomplete()) {
            // Handle command selection for access command
            if (interaction.commandName === 'access' && interaction.options.getFocused(true).name === 'command') {
                const focusedValue = interaction.options.getFocused().toString().toLowerCase();
                const commands = getCommandList();
                
                const filtered = commands.filter(choice => choice.toLowerCase().startsWith(focusedValue));
                
                await interaction.respond(
                    filtered.map(choice => ({ name: choice, value: choice })).slice(0, 25)
                );
            }
            return;
        }

        // Handle chat input commands
        if (interaction.isChatInputCommand()) {
            const command = client.commands.get(interaction.commandName);
            if (!command) {
                return interaction.reply({
                    flags: MessageFlags.Ephemeral,
                    content: "Comando desatualizado",
                });
            }
            
            // Initialize security service
            await SecurityService.initialize();
            
            // Skip security checks for access command
            if (interaction.commandName !== 'access') {
                // Check if command is public or user is admin
                const isPublic = await SecurityService.isCommandPublic(interaction.commandName);
                if (!isPublic && !SecurityService.isAdmin(interaction.user.id)) {
                    return interaction.reply({
                        flags: MessageFlags.Ephemeral,
                        content: "❌ Este comando é restrito apenas para administradores.",
                    });
                }
                
                // Check if user has exceeded rate limit
                // Use internal moderator check instead of Discord permissions
                if (!await SecurityService.checkRateLimit(interaction.user.id, interaction.commandName)) {
                    const remainingTime = Math.ceil((SecurityService.RATE_LIMIT_WINDOW / 1000) / 60);
                    return interaction.reply({
                        flags: MessageFlags.Ephemeral,
                        content: `❌ Você atingiu o limite de uso para este comando. Por favor, aguarde ${remainingTime} minuto(s) antes de tentar novamente.`,
                    });
                }
            }
            
            // Execute command
            command.execute(interaction, client);
        }
    },
};

/**
 * Get the list of available commands
 */
function getCommandList(): string[] {
    const commands: string[] = [];
    const commandsPath = join(process.cwd(), 'src', 'commands');
    
    // Loop through all category folders
    const categories = readdirSync(commandsPath, { withFileTypes: true })
        .filter(dirent => dirent.isDirectory())
        .map(dirent => dirent.name);
        
    for (const category of categories) {
        const categoryPath = join(commandsPath, category);
        
        // Get all command files
        const commandFiles = readdirSync(categoryPath)
            .filter(file => file.endsWith('.ts') || file.endsWith('.js'));
            
        for (const file of commandFiles) {
            commands.push(file.split('.')[0]);
        }
    }
    
    return commands;
}
