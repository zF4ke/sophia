import fs from "node:fs";
import { spawnSync } from "node:child_process";

if (fs.existsSync('.env')) process.loadEnvFile('.env');
const { readModelProfiles } = require('../src/app/modelProfiles') as typeof import('../src/app/modelProfiles');
const { SettingsService } = require('../src/app/SettingsService') as typeof import('../src/app/SettingsService');
let failed = false;
function check(name: string, ok: boolean, detail: string) { console.log(`${ok ? 'OK' : 'NEEDS ATTENTION'}  ${name}: ${detail}`); if (!ok) failed = true; }
const [nodeMajor, nodeMinor] = process.versions.node.split('.').map(Number);
check('Node', nodeMajor > 20 || (nodeMajor === 20 && nodeMinor >= 12), process.versions.node);
for (const key of ['DISCORD_TOKEN', 'OPENROUTER_API_KEY']) check(key, Boolean(process.env[key]), process.env[key] ? 'configured; value hidden' : 'add to .env');
try {
    const config = readModelProfiles();
    const settings = SettingsService.load();
    const profile = config.profiles[settings.modelProfile] ?? config.profiles[config.defaultProfile];
    check('Model profile', true, `${profile.label ?? profile.chatModel}; ${profile.api ?? 'chat-completions'}`);
    if (profile.apiKeyEnv) check(profile.apiKeyEnv, Boolean(process.env[profile.apiKeyEnv]), process.env[profile.apiKeyEnv] ? 'configured; value hidden' : 'add to .env');
    const docker = spawnSync('docker', ['version', '--format', '{{.Server.Os}}'], { encoding: 'utf8', windowsHide: true, timeout: 15000 });
    check('Docker engine', docker.status === 0 && docker.stdout.trim() === 'linux', docker.status === 0 ? docker.stdout.trim() : 'start Docker Desktop with Linux containers, then rerun');
    if (docker.status === 0) {
        const image = spawnSync('docker', ['image', 'inspect', settings.sandbox.image], { encoding: 'utf8', windowsHide: true, timeout: 15000 });
        check('Sandbox image', image.status === 0, image.status === 0 ? settings.sandbox.image : 'run npm run sandbox:build');
    }
    check('Availability', settings.guildAllowlist.length > 0 || settings.access.directMessages, 'use /access to enable a guild or DMs and grant authenticated users');
} catch (error) { check('Configuration', false, error instanceof Error ? error.message : 'invalid configuration'); }
console.log('Doctor does not send messages, call a model, repair Docker, or change access grants.');
process.exitCode = failed ? 1 : 0;
