import { AttachmentBuilder, ChatInputCommandInteraction, MessageFlags, SlashCommandBuilder } from "discord.js";
import { taskStore, type TaskRecord } from "@/runtime/tasks/TaskStore";
import { assertReadableChannels } from "@/security/SourceAccess";
import { readableToolRecords } from "@/runtime/sourceEvidence";
import { ActionVerifier } from "@/runtime/tasks/ActionVerifier";
import { AccessPolicy } from "@/security/AccessPolicy";
import { forgetTask } from "@/runtime/tasks/TaskRetention";

const labels: Record<TaskRecord["status"], string> = {
    running: "Em curso", completed: "Concluído", paused: "Pausado", cancelled: "Interrompido", failed: "Falhou",
};

export = {
    data: new SlashCommandBuilder().setName("tasks").setDescription("Mostra os teus pedidos recentes neste canal")
        .addBooleanOption(option => option.setName("ephemeral").setDescription("Mostrar só para ti"))
        .addStringOption(option => option.setName("task_id").setDescription("ID do pedido para consultar o plano, notas e registo de ações"))
        .addBooleanOption(option => option.setName("verify").setDescription("Verificar uma ação incerta no Discord sem a repetir"))
        .addBooleanOption(option => option.setName("forget").setDescription("Apagar os dados de um pedido inativo: notas, provas, ficheiros e registos"))
        .addUserOption(option => option.setName("collaborator").setDescription("Permitir que este utilizador corrija a direção de um pedido público"))
        .addBooleanOption(option => option.setName("remove_collaborator").setDescription("Remover o colaborador indicado"))
        .addStringOption(option => option.setName("file_path").setDescription("Ficheiro do pedido para descarregar"))
        .addStringOption(option => option.setName("action_id").setDescription("Ação incerta que verificaste no Discord"))
        .addStringOption(option => option.setName("resolution").setDescription("Resultado que confirmaste pessoalmente")
            .addChoices({ name: "A ação foi aplicada", value: "applied" }, { name: "A ação não foi aplicada", value: "not_applied" }))
        .addStringOption(option => option.setName("verification").setDescription("O que verificaste e onde; inclui IDs ou links quando possível")),
    async execute(interaction: ChatInputCommandInteraction) {
        const taskId = interaction.options.getString("task_id");
        const privateResponse = Boolean(interaction.options.getBoolean("ephemeral"));
        await interaction.deferReply({ flags: privateResponse ? MessageFlags.Ephemeral : undefined });
        const actionId = interaction.options.getString("action_id");
        const resolution = interaction.options.getString("resolution");
        const verification = interaction.options.getString("verification");
        const collaborator = interaction.options.getUser?.("collaborator");
        if (collaborator || interaction.options.getBoolean("remove_collaborator")) {
            if (!collaborator || !taskId || !interaction.guild || actionId || resolution || verification || interaction.options.getBoolean("forget") || interaction.options.getBoolean("verify") || interaction.options.getString("file_path")) {
                await interaction.editReply({ content: "Indica task_id e collaborator num pedido público deste servidor. Usa remove_collaborator:true para remover." }); return;
            }
            try {
                const remove = interaction.options.getBoolean("remove_collaborator") ?? false;
                if (!remove && await AccessPolicy.decide(collaborator.id, interaction.guild, "none") !== "allow") throw new Error("O colaborador precisa de autorização para usar Sophia neste servidor.");
                await taskStore.setCollaborator(taskId, interaction.user.id, interaction.channelId, interaction.guildId, collaborator.id, remove);
                await interaction.editReply({ content: `Colaborador ${collaborator.id} ${remove ? "removido" : "adicionado"} no pedido ${taskId}. Usa /steer com task_id. Alterações após uma correção de colaborador pedem aprovação ao dono do pedido.`, allowedMentions: { parse: [] } });
            } catch (error) { await interaction.editReply({ content: error instanceof Error ? error.message : "Não consegui alterar o colaborador.", allowedMentions: { parse: [] } }); }
            return;
        }
        if (interaction.options.getBoolean("forget")) {
            if (!taskId || actionId || resolution || verification || interaction.options.getBoolean("verify") || interaction.options.getString("file_path")) {
                await interaction.editReply({ content: "Para apagar os dados de um pedido inativo, indica apenas task_id e forget:true." }); return;
            }
            try {
                await forgetTask(taskId, interaction.user.id, interaction.channelId, interaction.guildId);
                await interaction.editReply({ content: `Dados do pedido ${taskId} apagados. Memórias duradouras, procedimentos aprovados e cópias de segurança têm gestão separada.`, allowedMentions: { parse: [] } });
            } catch (error) { await interaction.editReply({ content: error instanceof Error ? error.message : "Não consegui apagar os dados do pedido.", allowedMentions: { parse: [] } }); }
            return;
        }
        if (interaction.options.getBoolean("verify")) {
            if (!taskId || !actionId || resolution || verification) { await interaction.editReply({ content: "Para verificar sem repetir, indica apenas task_id, action_id e verify:true." }); return; }
            try {
                const observed = await ActionVerifier.verify({ actorId: interaction.user.id, guild: interaction.guild, currentChannelId: interaction.channelId, question: "Verify action", authorize: AccessPolicy.forActor(interaction.user.id, interaction.guild) }, taskId, actionId);
                const state = observed.resolution === "applied" ? "resultado confirmado" : observed.resolution === "not_applied" ? "alvo ainda presente" : "resultado ainda incerto";
                await interaction.editReply({ content: `Ação ${actionId}: ${state}. ${observed.detail}\nO pedido continua pausado.`, allowedMentions: { parse: [] } });
            } catch (error) { await interaction.editReply({ content: error instanceof Error ? error.message : "Não consegui verificar a ação.", allowedMentions: { parse: [] } }); }
            return;
        }
        if (actionId || resolution || verification) {
            if (!taskId || !actionId || !["applied", "not_applied"].includes(resolution ?? "") || !verification) {
                await interaction.editReply({ content: "Para resolver uma ação incerta, indica task_id, action_id, resolution e verification após conferires o resultado no Discord." });
                return;
            }
            try {
                await taskStore.resolveAction({ taskId, actionId, actorId: interaction.user.id, channelId: interaction.channelId,
                    guildId: interaction.guildId, resolution: resolution as "applied" | "not_applied", verification });
                await interaction.editReply({ content: `Ação ${actionId}: verificação registada por ti. O pedido continua pausado; retoma com /talk task_id:${taskId}.`, allowedMentions: { parse: [] } });
            } catch (error) {
                await interaction.editReply({ content: error instanceof Error ? error.message : "Não consegui guardar a verificação.", allowedMentions: { parse: [] } });
            }
            return;
        }
        if (taskId) {
            try { await assertReadableChannels(interaction.guild, interaction.user.id, await taskStore.evidenceChannels(taskId, interaction.user.id, interaction.channelId, interaction.guildId), { client: interaction.client, privateResponse, destinationChannelId: interaction.channelId }); }
            catch { await interaction.editReply({ content: "Já não tens acesso a uma das fontes deste pedido. O conteúdo derivado está indisponível." }); return; }
            const snapshot = await taskStore.snapshot(taskId, interaction.user.id, interaction.channelId, interaction.guildId);
            if (!snapshot) {
                await interaction.editReply({ content: "Não encontrei esse pedido entre os teus pedidos neste canal." });
                return;
            }
            const workspaceFiles = await taskStore.files(taskId, interaction.user.id, interaction.channelId, interaction.guildId);
            const filePath = interaction.options.getString("file_path");
            if (filePath) {
                const file = workspaceFiles.find(file => file.path === filePath);
                if (!file) { await interaction.editReply({ content: "Não encontrei esse ficheiro neste pedido." }); return; }
                try { await assertReadableChannels(interaction.guild, interaction.user.id, file.sourceChannelIds ?? [], { client: interaction.client, privateResponse, destinationChannelId: interaction.channelId }); }
                catch { await interaction.editReply({ content: "Já não tens acesso a uma das fontes deste ficheiro." }); return; }
                const bytes = Buffer.from(file.data, "base64");
                if (bytes.length > 8 * 1024 * 1024) { await interaction.editReply({ content: "O ficheiro excede 8 MiB. Retoma o pedido para o dividir ou comprimir." }); return; }
                await interaction.editReply({ files: [new AttachmentBuilder(bytes, { name: file.path.split("/").at(-1)! })], allowedMentions: { parse: [] } });
                return;
            }
            const toolRuns = await readableToolRecords(await taskStore.toolRuns(taskId, interaction.user.id, interaction.channelId, interaction.guildId), interaction.guild, interaction.user.id, { client: interaction.client, privateResponse, destinationChannelId: interaction.channelId });
            const approvals = await taskStore.approvals(taskId, interaction.user.id, interaction.channelId, interaction.guildId);
            const steering = await taskStore.steering(taskId, interaction.user.id, interaction.channelId, interaction.guildId);
            const usage = await taskStore.usage(taskId, interaction.user.id, interaction.channelId, interaction.guildId);
            const approvalMode = await taskStore.approvalMode(taskId, interaction.user.id);
            const collaborators = await taskStore.collaborators(taskId, interaction.user.id, interaction.channelId, interaction.guildId);
            const text = [`Task ${taskId}`, `Approval mode: ${approvalMode}`, `Collaborators (steering only): ${collaborators.join(", ") || "none"}`, "", "Plan", snapshot.plan ?? "No plan saved.", "", "Goals",
                ...snapshot.goals.map(goal => `#${goal.seq} [${goal.status}] ${goal.body}`), "", "Notes",
                ...snapshot.notes.map(note => `#${note.seq}${note.label ? ` [${note.label}]` : ""}\n${note.body}`), "", "Actions",
                ...snapshot.actions.map(action => JSON.stringify(action, null, 2)), "", "Approvals", JSON.stringify(approvals, null, 2),
                "", "Corrections", ...steering, "", "Model usage (provider attempts, including background work; configured-price estimates, not billing receipts; null means unavailable)", JSON.stringify(usage, null, 2),
                "", "Files", ...workspaceFiles.map(file => `${file.path} (${Buffer.byteLength(file.data, "base64")} bytes)`) ].join("\n");
            await interaction.editReply({ content: `Objetivos: ${snapshot.goals.length} · Notas: ${snapshot.notes.length} · Ações: ${snapshot.actions.length}`,
                files: [new AttachmentBuilder(Buffer.from(text, "utf8"), { name: "task-notes.txt" }),
                    ...toolRuns.length ? [new AttachmentBuilder(Buffer.from(JSON.stringify(toolRuns.reverse(), null, 2), "utf8"), { name: "task-evidence.json" })] : []],
                allowedMentions: { parse: [] } });
            return;
        }
        const candidates = await taskStore.list(interaction.user.id, interaction.channelId, interaction.guildId);
        const tasks = [];
        for (const task of candidates) {
            if (!privateResponse && await taskStore.privateOnly(task.id, interaction.user.id)) continue;
            try {
                await assertReadableChannels(interaction.guild, interaction.user.id, await taskStore.evidenceChannels(task.id, interaction.user.id, interaction.channelId, interaction.guildId), { client: interaction.client, privateResponse, destinationChannelId: interaction.channelId });
                tasks.push(task);
            } catch { /* Do not publish tasks whose sources cannot be shared here. */ }
        }
        const content = tasks.length ? tasks.map(task => {
            const objective = task.objective.replace(/[\r\n`*_~|<>]/g, " ").slice(0, 95);
            const status = task.reason === "interrupted" ? "Pausado após reinício" : labels[task.status];
            return `${status} · ${objective}\nID: ${task.id}`;
        }).join("\n\n") : "Não tens pedidos registados neste canal.";
        await interaction.editReply({ content, allowedMentions: { parse: [] } });
    },
};
