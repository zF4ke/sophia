import fs from "node:fs";
import path from "node:path";
import { ALL_TOOLS } from "../src/tools/registry";
import { DISCORD_TOOL_NAMES } from "../src/shared/discordTools";
import { discoverModuleFiles } from "../src/discord/loaders/moduleDiscovery";
import { AppPaths } from "../src/app/AppPaths";
import { SettingsService } from "../src/app/SettingsService";

const root = path.resolve(__dirname, "..");
const destination = path.join(root, "website/src/content/docs/reference");
const groups: Record<string, { title: string; guide: string; tools: string[] }> = {
    research: { title: "Discord and web research", guide: "research", tools: ["retrieve_messages", "search_messages", "random_channel_message", "list_guild_structure", "get_guild_context", "resolve_channel_targets", "resolve_member_identity", "get_member_profile", "list_members", "get_role_info", "list_roles", "list_threads", "read_thread_messages", "web_search", "fetch_url", "source_read", "index_channel", "corpus_create", "corpus_collect", "corpus_read"] },
    actions: { title: "Server actions and permissions", guide: "permissions", tools: ["create_channel", "create_category", "create_thread", "move_channel", "move_category", "manage_member_roles", "send_message", "edit_message", "create_role", "clear_messages", "delete_messages", "delete_channel", "delete_role", "edit_channel", "edit_role"] },
    tasks: { title: "Tasks and runtime inspection", guide: "tasks", tools: ["task_search", "task_control", "start_long_task", "note_add", "note_list", "note_clear", "plan_update", "goal_open", "goal_update", "goal_done", "inspect_runtime", "verify_action", "task_forget", "tool_search"] },
    memory: { title: "Memory and procedural skills", guide: "memory", tools: ["memory_search", "memory_remember", "memory_update", "memory_forget", "skill_search", "skill_load", "skill_save", "skill_evaluate", "skill_delete", "workflow_create", "workflow_list", "workflow_run", "workflow_delete"] },
    workspace: { title: "Workspace, media and calculation", guide: "workspace", tools: ["sandbox_run", "sandbox_import", "sandbox_publish", "sandbox_inspect", "sandbox_transcribe", "evaluate_math", "measure_text_length"] },
    cards: { title: "Cards and polls", guide: "cards", tools: ["artifact_send", "artifact_edit", "artifact_read", "create_poll", "get_poll_results"] },
    schedules: { title: "Scheduling", guide: "schedules", tools: ["schedule_create", "schedule_update", "schedule_list", "schedule_cancel"] },
};
const mapped = Object.values(groups).flatMap(g => g.tools);
if (new Set(mapped).size !== mapped.length || DISCORD_TOOL_NAMES.some(n => !mapped.includes(n)) || mapped.some(n => !DISCORD_TOOL_NAMES.includes(n as never))) throw new Error("Capability documentation coverage is incomplete or duplicated.");
fs.mkdirSync(path.join(destination, "tools"), { recursive: true });
for (const folder of ['tools', 'command-options']) {
    const directory = path.join(destination, folder);
    fs.mkdirSync(directory, { recursive: true });
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
        if (entry.isFile() && entry.name.endsWith('.md')) fs.unlinkSync(path.join(directory, entry.name));
    }
}
const escape = (text: string) => text.replace(/\|/g, "\\|").replace(/\r?\n/g, " ");
let index = '---\ntitle: Every feature\nsidebar:\n  order: 1\n---\n\nThis catalogue is checked against the runtime capability list at build time. Each tool links to its current schema and a user guide. Tool availability still depends on admission, permissions and enabled services.\n\n';
for (const group of Object.values(groups)) {
    const guide = `../../../use/${group.guide}/`;
    index += `## ${group.title}\n\n[Read the guide](../../use/${group.guide}/).\n\n| Capability | Effect | Purpose |\n| --- | --- | --- |\n`;
    for (const name of group.tools) {
        const tool = ALL_TOOLS.find(t => t.name === name);
        if (!tool) throw new Error(`Missing registered tool ${name}`);
        index += `| [\`${name}\`](../tools/${name}/) | ${tool.catalog.effect} | ${escape(tool.catalog.description)} |\n`;
        fs.writeFileSync(path.join(destination, "tools", `${name}.md`), `---\ntitle: ${name}\n---\n\n${tool.schema.description}\n\n[User guide](${guide}) / [All capabilities](../../features/)\n\n## Contract\n\nEffect: **${tool.catalog.effect}**. Evidence role: \`${tool.catalog.evidenceRole}\`.\n\n${tool.capability.description}\n\n## Parameters\n\nThis is the current provider-facing JSON Schema, generated from the runtime definition.\n\n\`\`\`json\n${JSON.stringify(tool.schema.parameters, null, 2)}\n\`\`\`\n\n## Execution requirements\n\n${[...tool.capability.authRequirements, ...tool.capability.preconditions].map(x => `- ${x}`).join('\n') || 'The ordinary admission and tool execution checks apply.'}\n\n## Expected postconditions\n\n${tool.capability.postconditions.map(x => `- ${x}`).join('\n') || 'Inspect the returned status, summary and data; do not infer success from a tool call alone.'}\n`);
    }
    index += '\n';
}
index += '## Features beyond tools\n\n| Feature | Guide |\n| --- | --- |\n| Mentions, replies, attachments and one identity | [Conversation](../../use/conversation/) |\n| Authenticated grants, approval rules and protected resources | [Permissions](../../use/permissions/) |\n| Durable resume, steering, cancellation, collaborators and handoff | [Tasks](../../use/tasks/) |\n| Idle dreaming, scoped recall and forgetting | [Memory](../../use/memory/) |\n| Provider attempts and installation costs | [Costs](../../use/costs/) |\n| Models, voice and settings | [Setup](../../setup/models/) and [commands](../commands/) |\n| Backups, migration and another PC | [Move installation](../../setup/another-pc/) |\n| Developer diagnostics and index repair | [Command guide](../../build/commands-and-admin/) |\n';
fs.writeFileSync(path.join(destination, "features.md"), index);
let commands = '---\ntitle: Command options\nsidebar:\n  order: 2\n---\n\nGenerated from the registered command builders. Open a command for its actual options. See the [user guides](../../use/conversation/) for examples and [operator guide](../../build/commands-and-admin/) for access requirements.\n\n| Command | Purpose |\n| --- | --- |\n';
fs.mkdirSync(path.join(destination, 'command-options'), { recursive: true });
for (const file of discoverModuleFiles(AppPaths.commandModulesRoot, '.command')) {
    const loaded = require(file);
    const command = loaded.default ?? loaded;
    if (!command?.data?.name || typeof command.execute !== 'function') continue;
    const data = command.data.toJSON();
    commands += `| [/${data.name}](../command-options/command-${data.name}/) | ${escape(data.description)} |\n`;
    fs.writeFileSync(path.join(destination, 'command-options', `command-${data.name}.md`), `---\ntitle: /${data.name}\n---\n\n${data.description}\n\n[All commands](../../commands/) / [Usage and permissions](../../../build/commands-and-admin/)\n\n## Options\n\n${data.options?.length ? `\`\`\`json\n${JSON.stringify(data.options, null, 2)}\n\`\`\`` : 'This command takes no options.'}\n`);
}
fs.writeFileSync(path.join(destination, 'commands.md'), commands);
fs.writeFileSync(path.join(destination, 'configuration.md'), `---\ntitle: Configuration defaults\n---\n\nGenerated from SettingsService defaults, never from a running installation's settings. Secrets belong in .env. Operator settings belong in storage/settings.json.\n\n## Groups\n\n| Group | Responsibility |\n| --- | --- |\n| modelProfile | Main model selection; see the model guide. |\n| runtime | Context, retrieval, concurrency, approvals and explicit optional limits. |\n| compaction | Summarizer selection and context thresholds. |\n| memory | Idle dreaming and its interval. |\n| sandbox | Container execution switch and image. |\n| scheduling | Scheduler switch and polling interval. |\n| access, guildAllowlist | Authenticated grants, rules and availability. |\n| protectedChannelIds | Channels protected from destructive actions. |\n| voice | Shared balanced, casual or formal presentation. |\n| debug | Diagnostic logging. |\n\n## Fresh-install defaults\n\n\`\`\`json\n${JSON.stringify({ ...SettingsService.getDefaults(), runtime: { ...SettingsService.getDefaults().runtime, operationalDbPath: 'storage/runtime/operational.sqlite' } }, null, 2)}\n\`\`\`\n\nThe operational path above is illustrative. Sophia resolves its actual default under the chosen storage root. Do not copy an absolute path from another PC. Use /settings and /access for validated changes.\n`);
console.log(`Documented ${mapped.length} capabilities and all registered commands.`);
