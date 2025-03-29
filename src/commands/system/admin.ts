import { SlashCommandBuilder, ChatInputCommandInteraction, PermissionFlagsBits } from "discord.js";
import { SecurityService } from "../../services/SecurityService";

export default {
    data: new SlashCommandBuilder()
        .setName('admin')
        .setDescription('Comandos de administração para gerenciar a segurança do bot')
        .setDefaultMemberPermissions(PermissionFlagsBits.Administrator)
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
        ),
    async execute(interaction: ChatInputCommandInteraction) {
        // Inicializar serviço de segurança
        await SecurityService.initialize();
        
        // Verificar se o usuário é um administrador
        if (!SecurityService.isAdmin(interaction.user.id)) {
            await interaction.reply({ content: '❌ Você não tem permissão para usar este comando.', ephemeral: true });
            return;
        }

        const group = interaction.options.getSubcommandGroup();
        const subcommand = interaction.options.getSubcommand();
        
        // Gerenciar administradores
        if (group === 'admins') {
            if (subcommand === 'list') {
                const admins = await SecurityService.getAllAdmins();
                
                if (admins.length === 0) {
                    await interaction.reply({ content: 'Nenhum administrador encontrado', ephemeral: true });
                    return;
                }
                
                const adminList = admins.map(admin => {
                    const date = new Date(admin.addedAt).toLocaleString();
                    return `• <@${admin.userId}> (Adicionado por: ${admin.addedBy === 'system' ? 'Sistema' : `<@${admin.addedBy}>`} em ${date})`;
                }).join('\n');
                
                await interaction.reply({ 
                    content: `### Usuários Administradores\n${adminList}`, 
                    ephemeral: true,
                    allowedMentions: { parse: [] }
                });
            }
            else if (subcommand === 'add') {
                const user = interaction.options.getUser('user');
                if (!user) {
                    await interaction.reply({ content: '❌ Usuário não encontrado', ephemeral: true });
                    return;
                }
                
                const added = await SecurityService.addAdmin(user.id, interaction.user.id);
                
                if (added) {
                    await interaction.reply({ 
                        content: `✅ <@${user.id}> foi adicionado como administrador.`, 
                        ephemeral: true,
                        allowedMentions: { parse: [] }
                    });
                } else {
                    await interaction.reply({ 
                        content: `❌ <@${user.id}> já é um administrador.`, 
                        ephemeral: true,
                        allowedMentions: { parse: [] }
                    });
                }
            }
            else if (subcommand === 'remove') {
                const user = interaction.options.getUser('user');
                if (!user) {
                    await interaction.reply({ content: '❌ Usuário não encontrado', ephemeral: true });
                    return;
                }
                
                const removed = await SecurityService.removeAdmin(user.id);
                
                if (removed) {
                    await interaction.reply({ 
                        content: `✅ <@${user.id}> foi removido dos administradores.`, 
                        ephemeral: true,
                        allowedMentions: { parse: [] }
                    });
                } else {
                    await interaction.reply({ 
                        content: `❌ <@${user.id}> não pôde ser removido. Pode ser um administrador fixo ou não é um administrador.`, 
                        ephemeral: true,
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
                    await interaction.reply({ content: 'Nenhum moderador encontrado', ephemeral: true });
                    return;
                }
                
                const moderatorList = moderators.map(mod => {
                    const date = new Date(mod.addedAt).toLocaleString();
                    return `• <@${mod.userId}> (Adicionado por: ${mod.addedBy === 'system' ? 'Sistema' : `<@${mod.addedBy}>`} em ${date})`;
                }).join('\n');
                
                await interaction.reply({ 
                    content: `### Usuários Moderadores\n${moderatorList}`, 
                    ephemeral: true,
                    allowedMentions: { parse: [] }
                });
            }
            else if (subcommand === 'add') {
                const user = interaction.options.getUser('user');
                if (!user) {
                    await interaction.reply({ content: '❌ Usuário não encontrado', ephemeral: true });
                    return;
                }
                
                const added = await SecurityService.addModerator(user.id, interaction.user.id);
                
                if (added) {
                    await interaction.reply({ 
                        content: `✅ <@${user.id}> foi adicionado como moderador.`, 
                        ephemeral: true,
                        allowedMentions: { parse: [] }
                    });
                } else {
                    await interaction.reply({ 
                        content: `❌ <@${user.id}> já é um moderador ou administrador.`, 
                        ephemeral: true,
                        allowedMentions: { parse: [] }
                    });
                }
            }
            else if (subcommand === 'remove') {
                const user = interaction.options.getUser('user');
                if (!user) {
                    await interaction.reply({ content: '❌ Usuário não encontrado', ephemeral: true });
                    return;
                }
                
                const removed = await SecurityService.removeModerator(user.id);
                
                if (removed) {
                    await interaction.reply({ 
                        content: `✅ <@${user.id}> foi removido dos moderadores.`, 
                        ephemeral: true,
                        allowedMentions: { parse: [] }
                    });
                } else {
                    await interaction.reply({ 
                        content: `❌ <@${user.id}> não é um moderador.`, 
                        ephemeral: true,
                        allowedMentions: { parse: [] }
                    });
                }
            }
        }
        // Gerenciar comandos
        else if (group === 'command') {
            const commandName = interaction.options.getString('command');
            if (!commandName) {
                await interaction.reply({ content: '❌ Nome do comando não fornecido', ephemeral: true });
                return;
            }
            
            if (subcommand === 'visibility') {
                const isPublic = interaction.options.getBoolean('public') ?? true;
                
                await SecurityService.setCommandVisibility(commandName, isPublic);
                
                await interaction.reply({ 
                    content: `✅ Comando \`${commandName}\` agora é ${isPublic ? 'público' : 'privado'}.`, 
                    ephemeral: true 
                });
            }
            else if (subcommand === 'limit') {
                const defaultLimit = interaction.options.getInteger('default') ?? 5;
                const adminLimit = interaction.options.getInteger('admin') ?? defaultLimit * 2;
                const moderatorLimit = interaction.options.getInteger('moderator') ?? Math.floor(defaultLimit * 1.5);
                
                await SecurityService.setCommandRateLimit(commandName, defaultLimit, adminLimit, moderatorLimit);
                
                await interaction.reply({ 
                    content: `✅ Limites para o comando \`${commandName}\` foram definidos:\n• Padrão: ${defaultLimit}\n• Moderador: ${moderatorLimit}\n• Admin: ${adminLimit}`, 
                    ephemeral: true 
                });
            }
        }
    }
};