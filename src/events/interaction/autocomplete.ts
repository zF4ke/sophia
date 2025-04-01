import { AutocompleteInteraction, Events } from "discord.js";
import { readdirSync } from "fs";
import { join } from "path";

export default {
    name: Events.InteractionCreate,
    async execute(interaction: AutocompleteInteraction) {
        if (!interaction.isAutocomplete()) return;

        console.log(`Autocomplete interaction: ${interaction.commandName}`);

        // Handle command selection for access command
        if (interaction.commandName === 'access' && interaction.options.getFocused(true).name === 'command') {
            const focusedValue = interaction.options.getFocused().toString().toLowerCase();
            const commands = getCommandList();

            console.log(`Focused value: ${focusedValue}`);
            console.log(`Commands: ${commands}`);
            
            const filtered = commands.filter(choice => choice.toLowerCase().startsWith(focusedValue));
            
            await interaction.respond(
                filtered.map(choice => ({ name: choice, value: choice })).slice(0, 25)
            );
        }
    }
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