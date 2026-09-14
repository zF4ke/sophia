import { afterEach, expect, it, vi } from "vitest";
import { MessageFlags } from "discord.js";
import settingsCommand from "@/discord/commands/system/settings/settings.command";
import costsCommand from "@/discord/commands/tools/costs.command";
import tasksCommand from "@/discord/commands/tools/tasks.command";
import stopCommand from "@/discord/commands/tools/stop.command";
import { handleSettingsPanelInteraction } from "@/discord/commands/system/settings/settingsInteractions";
import { SecurityService } from "@/security/SecurityService";
import { taskStore } from "@/runtime/tasks/TaskStore";
vi.mock("@/discord/commands/system/costsPanel", () => ({ buildCostsPanel: vi.fn(async () => ({ components: [], flags: MessageFlags.IsComponentsV2 })) }));
afterEach(() => vi.restoreAllMocks());
function input(privateReply = false) { return { user: { id: "owner" }, channelId: "channel", guildId: null, guild: null,
    options: { getBoolean: () => privateReply, getString: () => null }, reply: vi.fn(), deferReply: vi.fn(), editReply: vi.fn(), deferUpdate: vi.fn() }; }
it("makes settings, own costs and stop public by default with an explicit private option", async () => {
    vi.spyOn(SecurityService, "initialize").mockResolvedValue();
    vi.spyOn(SecurityService, "isAdmin").mockReturnValue(true);
    for (const ephemeral of [false, true]) {
        const interaction = input(ephemeral);
        await settingsCommand.execute(interaction as never, {} as never);
        expect(interaction.reply.mock.calls[0][0].flags & MessageFlags.Ephemeral).toBe(ephemeral ? 64 : 0);
        await costsCommand.execute(interaction as never);
        expect(Boolean(interaction.deferReply.mock.calls[0][0].flags)).toBe(ephemeral);
        interaction.reply.mockClear();
        await stopCommand.execute(interaction as never);
        expect(Boolean(interaction.reply.mock.calls[0][0].flags)).toBe(ephemeral);
    }
});
it("updates the settings message with installation-wide costs", async () => {
    vi.spyOn(SecurityService, "initialize").mockResolvedValue();
    vi.spyOn(SecurityService, "isAdmin").mockReturnValue(true);
    const interaction = { ...input(), customId: "settings:tab:costs" };
    await handleSettingsPanelInteraction(interaction as never);
    expect(interaction.deferUpdate).toHaveBeenCalled();
    expect(interaction.deferReply).not.toHaveBeenCalled();
});
it("omits private task objectives from public summaries", async () => {
    vi.spyOn(taskStore, "list").mockResolvedValue([{ id: "private", objective: "Private medical question", status: "completed" }, { id: "public", objective: "Public event review", status: "completed" }] as never);
    vi.spyOn(taskStore, "privateOnly").mockImplementation(async id => id === "private");
    vi.spyOn(taskStore, "evidenceChannels").mockResolvedValue([]);
    const interaction = input();
    await tasksCommand.execute(interaction as never);
    const content = interaction.editReply.mock.calls[0][0].content;
    expect(content).toContain("Public event review");
    expect(content).not.toContain("medical");
    expect(interaction.deferReply).toHaveBeenCalledWith({ flags: undefined });
});

 it("shows skill, memory and schedule records publicly unless a private reply is requested", async () => {
    const { default: skills } = await import("@/discord/commands/tools/skills.command");
    const { default: memories } = await import("@/discord/commands/tools/memories.command");
    const { default: schedules } = await import("@/discord/commands/tools/schedules.command");
    const { SkillStore } = await import("@/memory/SkillStore");
    const { knowledgeStore } = await import("@/memory/KnowledgeStore");
    const { scheduleStore } = await import("@/runtime/scheduling/ScheduleStore");
    vi.spyOn(SkillStore, "search").mockResolvedValue([]);
    vi.spyOn(knowledgeStore, "search").mockResolvedValue([]);
    vi.spyOn(scheduleStore, "list").mockResolvedValue([]);
    for (const ephemeral of [false, true]) {
        for (const command of [skills, memories, schedules]) {
            const interaction = input();
            interaction.options.getBoolean = (name?: string) => name === "ephemeral" && ephemeral;
            await command.execute(interaction as never);
            expect(interaction.deferReply).toHaveBeenCalledWith({ flags: ephemeral ? MessageFlags.Ephemeral : undefined });
            expect(interaction.editReply).toHaveBeenCalled();
        }
    }
});
