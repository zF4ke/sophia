import { SlashCommandBuilder, ChatInputCommandInteraction, PermissionFlagsBits, MessageFlags } from "discord.js";
import { SecurityService } from "../../services/SecurityService";

const DEFAULT_EPHEMERAL = false; // Change this to false if you want to disable ephemeral responses globally
const DEFAULT_LIMIT = 5;
const DEFAULT_ADMIN_LIMIT = 10;
const DEFAULT_MODERATOR_LIMIT = 7;

module.exports = {
    data: new SlashCommandBuilder()
        .setName('access') // Renamed from 'admin' to 'access'
        .setContexts(0, 1, 2) // Keep setContexts
        .setIntegrationTypes(1) // Keep setIntegrationTypes
        .setDescription('Comandos de administração para gerenciar a segurança do bot')
        .setDefaultMemberPermissions(PermissionFlagsBits.Administrator)
        .setDMPermission(false) // Hide from the command list
        .addSubcommandGroup(group => 
            group
                .setName('admins')
                .setDescription('Gerenciar usuários administradores')
                .addSubcommand(subcommand =>
                    subcommand
                        .setName('list')
                        .setDescription('Listar todos os administradores')
                )
                .addSubcommand(subcommand =>
                    subcommand
                        .setName('add')
                        .setDescription('Adicionar um novo administrador')
                        .addUserOption(option =>
                            option.setName('user')
                                .setDescription('O usuário para adicionar como administrador')
                                .setRequired(true)
                        )
                )
                .addSubcommand(subcommand =>
                    subcommand
                        .setName('remove')
                        .setDescription('Remover um administrador')
                        .addUserOption(option =>
                            option.setName('user')
                                .setDescription('O administrador para remover')
                                .setRequired(true)
                        )
                )
        )
        .addSubcommandGroup(group => 
            group
                .setName('moderators')
                .setDescription('Gerenciar usuários moderadores')
                .addSubcommand(subcommand =>
                    subcommand
                        .setName('list')
                        .setDescription('Listar todos os moderadores')
                )
                .addSubcommand(subcommand =>
                    subcommand
                        .setName('add')
                        .setDescription('Adicionar um novo moderador')
                        .addUserOption(option =>
                            option.setName('user')
                                .setDescription('O usuário para adicionar como moderador')
                                .setRequired(true)
                        )
                )
                .addSubcommand(subcommand =>
                    subcommand
                        .setName('remove')
                        .setDescription('Remover um moderador')
                        .addUserOption(option =>
                            option.setName('user')
                                .setDescription('O moderador para remover')
                                .setRequired(true)
                        )
                )
        )
        .addSubcommandGroup(group =>
            group
                .setName('command')
                .setDescription('Gerenciar configurações de comandos')
                .addSubcommand(subcommand =>
                    subcommand
                        .setName('visibility')
                        .setDescription('Definir visibilidade do comando')
                        .addStringOption(option =>
                            option.setName('command')
                                .setDescription('O comando a ser configurado')
                                .setRequired(true)
                                .setAutocomplete(true)
                        )
                        .addBooleanOption(option =>
                            option.setName('public')
                                .setDescription('Se o comando é público')
                                .setRequired(true)
                        )
                )
                .addSubcommand(subcommand =>
                    subcommand
                        .setName('limit')
                        .setDescription('Definir limites de uso do comando')
                        .addStringOption(option =>
                            option.setName('command')
                                .setDescription('O comando a ser configurado')
                                .setRequired(true)
                                .setAutocomplete(true)
                        )
                        .addIntegerOption(option =>
                            option.setName('default')
                                .setDescription('Limite padrão para usuários comuns')
                                .setRequired(true)
                                .setMinValue(1)
                                .setMaxValue(100)
                        )
                        .addIntegerOption(option =>
                            option.setName('admin')
                                .setDescription('Limite para administradores')
                                .setRequired(false)
                                .setMinValue(1)
                                .setMaxValue(200)
                        )
                        .addIntegerOption(option =>
                            option.setName('moderator')
                                .setDescription('Limite para moderadores')
                                .setRequired(false)
                                .setMinValue(1)
                                .setMaxValue(150)
                        )
                )
                .addSubcommand(subcommand =>
                    subcommand
                        .setName('list')
                        .setDescription('Listar todos os comandos')
                )
        ),
    async execute(interaction: ChatInputCommandInteraction) {
        // Inicializar serviço de segurança
        await SecurityService.initialize();

        const ephemeral = interaction.options.getBoolean("ephemeral") ?? DEFAULT_EPHEMERAL;
        
        // Verificar se o usuário é um administrador
        if (!SecurityService.isAdmin(interaction.user.id)) {
            await interaction.reply({ content: '❌ Você não tem permissão para usar este comando.', flags: ephemeral ? MessageFlags.Ephemeral : undefined });
            return;
        }

        const group = interaction.options.getSubcommandGroup();
        const subcommand = interaction.options.getSubcommand();
        
        // Gerenciar administradores
        if (group === 'admins') {
            if (subcommand === 'list') {
                const admins = await SecurityService.getAllAdmins();
                
                if (admins.length === 0) {
                    await interaction.reply({ content: 'Nenhum administrador encontrado', flags: ephemeral ? MessageFlags.Ephemeral : undefined });
                    return;
                }
                
                const adminList = await Promise.all(admins.map(async admin => {
                    const date = new Date(admin.addedAt).toLocaleString();
                    const username = await fetchUserUsername(interaction, admin.userId);

                    return `• **${username}** (Adicionado por: ${admin.addedBy === 'system' ? 'Sistema' : admin.addedBy} em ${date})`;
                }));
                
                await interaction.reply({ 
                    content: `### Usuários Administradores\n${adminList.join('\n')}`, 
                    flags: ephemeral ? MessageFlags.Ephemeral : undefined,
                    allowedMentions: { parse: [] }
                });
            }
            else if (subcommand === 'add') {
                const user = interaction.options.getUser('user');
                if (!user) {
                    await interaction.reply({ content: '❌ Usuário não encontrado', flags: ephemeral ? MessageFlags.Ephemeral : undefined });
                    return;
                }
                
                const added = await SecurityService.addAdmin(user.id, interaction.user.id);
                const username = user.username || user.id; // Fallback to user ID if username is not available
                
                if (added) {
                    await interaction.reply({ 
                        content: `✅ **${username}** foi adicionado como administrador.`, 
                        flags: ephemeral ? MessageFlags.Ephemeral : undefined,
                        allowedMentions: { parse: [] }
                    });
                } else {
                    await interaction.reply({ 
                        content: `❌ **${username}** já é um administrador.`, 
                        flags: ephemeral ? MessageFlags.Ephemeral : undefined,
                        allowedMentions: { parse: [] }
                    });
                }
            }
            else if (subcommand === 'remove') {
                const user = interaction.options.getUser('user');
                if (!user) {
                    await interaction.reply({ content: '❌ Usuário não encontrado', flags: ephemeral ? MessageFlags.Ephemeral : undefined });
                    return;
                }
                
                const removed = await SecurityService.removeAdmin(user.id);
                const username = user.username || user.id; // Fallback to user ID if username is not available

                if (removed) {
                    await interaction.reply({ 
                        content: `✅ **${username}** foi removido dos administradores.`, 
                        flags: ephemeral ? MessageFlags.Ephemeral : undefined,
                        allowedMentions: { parse: [] }
                    });
                } else {
                    await interaction.reply({ 
                        content: `❌ **${username}** não pôde ser removido. Pode ser um administrador fixo ou não é um administrador.`, 
                        flags: ephemeral ? MessageFlags.Ephemeral : undefined,
                        allowedMentions: { parse: [] }
                    });
                }
            }
        }
        // Gerenciar moderadores
        else if (group === 'moderators') {
            if (subcommand === 'list') {
                const moderators = await SecurityService.getAllModerators();
                
                if (moderators.length === 0) {
                    await interaction.reply({ content: 'Nenhum moderador encontrado', flags: ephemeral ? MessageFlags.Ephemeral : undefined });
                    return;
                }
                
                const moderatorList = await Promise.all(moderators.map(async mod => {
                    const date = new Date(mod.addedAt).toLocaleString();
                    const username = await fetchUserUsername(interaction, mod.userId);

                    return `• **${username}** (Adicionado por: ${mod.addedBy === 'system' ? 'Sistema' : `<@${mod.addedBy}>`} em ${date})`;
                }));
                
                await interaction.reply({ 
                    content: `### Usuários Moderadores\n${moderatorList.join('\n')}`, 
                    flags: ephemeral ? MessageFlags.Ephemeral : undefined,
                    allowedMentions: { parse: [] }
                });
            }
            else if (subcommand === 'add') {
                const user = interaction.options.getUser('user');
                if (!user) {
                    await interaction.reply({ content: '❌ Usuário não encontrado', flags: ephemeral ? MessageFlags.Ephemeral : undefined });
                    return;
                }
                
                const added = await SecurityService.addModerator(user.id, interaction.user.id);
                const username = user.username || user.id; // Fallback to user ID if username is not available
                
                if (added) {
                    await interaction.reply({ 
                        content: `✅ **${username}** foi adicionado como moderador.`, 
                        flags: ephemeral ? MessageFlags.Ephemeral : undefined,
                        allowedMentions: { parse: [] }
                    });
                } else {
                    await interaction.reply({ 
                        content: `❌ **${username}** já é um moderador ou administrador.`, 
                        flags: ephemeral ? MessageFlags.Ephemeral : undefined,
                        allowedMentions: { parse: [] }
                    });
                }
            }
            else if (subcommand === 'remove') {
                const user = interaction.options.getUser('user');
                if (!user) {
                    await interaction.reply({ content: '❌ Usuário não encontrado', flags: ephemeral ? MessageFlags.Ephemeral : undefined });
                    return;
                }
                
                const removed = await SecurityService.removeModerator(user.id);
                const username = user.username || user.id; // Fallback to user ID if username is not available
                
                if (removed) {
                    await interaction.reply({ 
                        content: `✅ **${username}** foi removido dos moderadores.`, 
                        flags: ephemeral ? MessageFlags.Ephemeral : undefined,
                        allowedMentions: { parse: [] }
                    });
                } else {
                    await interaction.reply({ 
                        content: `❌ **${username}** não é um moderador.`, 
                        flags: ephemeral ? MessageFlags.Ephemeral : undefined,
                        allowedMentions: { parse: [] }
                    });
                }
            }
        }
        // Gerenciar comandos
        else if (group === 'command') {
            const commandName = interaction.options.getString('command');
            if (subcommand !== 'list') {
                if (!commandName) {
                    await interaction.reply({ content: '❌ Comando não encontrado', flags: ephemeral ? MessageFlags.Ephemeral : undefined });
                    return;
                }
                
                if (subcommand === 'visibility') {
                    const isPublic = interaction.options.getBoolean('public') ?? true;
                    
                    await SecurityService.setCommandVisibility(commandName, isPublic);
                    
                    await interaction.reply({ 
                        content: `✅ Comando \`${commandName}\` agora é ${isPublic ? 'público' : 'privado'}.`, 
                        flags: ephemeral ? MessageFlags.Ephemeral : undefined
                    });
                }
                else if (subcommand === 'limit') {
                    const defaultLimit = interaction.options.getInteger('default') ?? DEFAULT_LIMIT;
                    const adminLimit = interaction.options.getInteger('admin') ?? DEFAULT_ADMIN_LIMIT;
                    const moderatorLimit = interaction.options.getInteger('moderator') ?? DEFAULT_MODERATOR_LIMIT;
                    
                    await SecurityService.setCommandRateLimit(commandName, defaultLimit, adminLimit, moderatorLimit);
                    
                    await interaction.reply({ 
                        content: `✅ Limites para o comando \`${commandName}\` foram definidos:\n• Padrão: ${defaultLimit}\n• Moderador: ${moderatorLimit}\n• Admin: ${adminLimit}`, 
                        flags: ephemeral ? MessageFlags.Ephemeral : undefined
                    });
                }
            }
            else if (subcommand === 'list') {
                const commands = await SecurityService.getCommandConfigs();
                
                if (commands.size === 0) {
                    await interaction.reply({ content: 'Nenhum comando encontrado', flags: ephemeral ? MessageFlags.Ephemeral : undefined });
                    return;
                }
                
                // transform to array of objects
                /*
                    {
                        name: 'commandName', // the Map key
                        ...config // the Map value
                    }
                */
                const commandConfigs = Array.from(commands, ([name, config]) => ({ name, ...config }));
                const commandList = commandConfigs.map(cmd => {
                    const visibility = cmd.isPublic ? 'Público' : 'Privado';
                    const defaultLimit = cmd.rateLimits.default || DEFAULT_LIMIT;
                    const adminLimit = cmd.rateLimits.admin || DEFAULT_ADMIN_LIMIT;
                    const moderatorLimit = cmd.rateLimits.moderator || DEFAULT_MODERATOR_LIMIT;
                    return `• **${cmd.name}**: ${visibility} | Limites: Padrão: ${defaultLimit}, Moderador: ${moderatorLimit}, Admin: ${adminLimit}`;
                });
                
                await interaction.reply({ 
                    content: `### Comandos Disponíveis\n${commandList.join('\n')}`, 
                    flags: ephemeral ? MessageFlags.Ephemeral : undefined
                });
            }
        }
    }
};

async function fetchUserUsername(interaction: ChatInputCommandInteraction, userId: string) {
    //const user = await interaction.client.users.fetch(userId).catch(() => null);
    // try cache first
    let user = interaction.client.users.cache.get(userId) || null;
    if (user) return user.username;

    // try to fetch from API
    let fetchedUser = await interaction.client.users.fetch(userId).catch(() => null);
    if (fetchedUser) {
        interaction.client.users.cache.set(userId, fetchedUser);
        return fetchedUser.username;
    }

    return userId; // Fallback to userId if not found
}