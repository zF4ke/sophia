import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../', import.meta.url));
const output = path.join(root, 'src/content/docs');
const repo = path.resolve(root, '..');
await fs.mkdir(path.join(root, 'public'), { recursive: true });
await fs.copyFile(path.join(repo, 'assets/logo.svg'), path.join(root, 'public/icon.svg'));
const reference = new Set(['v5-plan', 'live-acceptance', 'feature-user-stories']);
await fs.mkdir(path.join(output, 'build'), { recursive: true });
await fs.mkdir(path.join(output, 'reference'), { recursive: true });
const names = (await fs.readdir(path.join(repo, 'docs'))).filter(n => n.endsWith('.md'));
for (const name of names) {
  const slug = name.slice(0, -3);
  let body = await fs.readFile(path.join(repo, 'docs', name), 'utf8');
  const title = /^# (.+)\r?$/m.exec(body)?.[1] || slug;
  body = body.replace(/^# .+\r?\n/, '');
  body = body.replace(/\]\((?:\.\/)?([\w-]+)\.md(#[^)]*)?\)/g, (_, target, hash = '') => `](../../${reference.has(target) ? 'reference' : 'build'}/${target}/${hash})`);
  body = body.replace(/\]\((?:\.\.\/)+([^)]*)\)/g, (match, target) => {
    if (match.includes('../../build/') || match.includes('../../reference/')) return match;
    return `](https://github.com/zF4ke/Sophia/blob/master/${target})`;
  });
  await fs.writeFile(path.join(output, reference.has(slug) ? 'reference' : 'build', name), `---\ntitle: ${JSON.stringify(title)}\n---\n\n${body}`);
}
console.log(`Synced ${names.length} canonical engineering documents. Edit docs/, not these generated copies.`);
