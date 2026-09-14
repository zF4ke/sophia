import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseHTML } from '../../node_modules/linkedom/esm/index.js';
const root = fileURLToPath(new URL('../dist/', import.meta.url));
const base = (process.env.DOCS_BASE || '/').replace(/\/$/, '');
async function files(dir) { return (await Promise.all((await fs.readdir(dir, { withFileTypes: true })).map(e => e.isDirectory() ? files(path.join(dir, e.name)) : [path.join(dir, e.name)]))).flat(); }
const pages = (await files(root)).filter(f => f.endsWith('.html'));
const documents = new Map();
for (const file of pages) documents.set(file, parseHTML(await fs.readFile(file, 'utf8')).document);
const failures = [];
for (const [file, doc] of documents) {
    const relative = path.relative(root, file).replaceAll('\\', '/').replace(/index\.html$/, '');
    const current = new URL(`${base}/${relative}`, 'https://docs.invalid');
    for (const resource of doc.querySelectorAll('img[src], script[src], link[rel="stylesheet"][href], link[rel="icon"][href]')) {
        const href = resource.getAttribute('src') || resource.getAttribute('href');
        const url = new URL(href, current);
        if (url.origin !== current.origin) continue;
        const pathname = decodeURIComponent(url.pathname);
        if (base && !pathname.startsWith(base + '/')) { failures.push(`${relative}: asset escapes deployment base: ${href}`); continue; }
        try { await fs.access(path.join(root, pathname.slice(base.length).replace(/^\//, ''))); }
        catch { failures.push(`${relative}: missing asset ${href}`); }
    }
    for (const chat of doc.querySelectorAll('.discord-example')) {
        if (!chat.querySelector('.chat-channel') || !chat.querySelector('.chat-message')) failures.push(`${relative}: empty conversation example`);
        for (const message of chat.querySelectorAll('.chat-message')) {
            if (!message.querySelector('.chat-avatar') || !message.querySelector('.chat-author') || !message.querySelector('.chat-message-body')?.textContent.trim()) failures.push(`${relative}: malformed conversation message`);
        }
    }
    for (const quote of doc.querySelectorAll('blockquote')) {
        if (/^(You|Sophia)\b/.test(quote.textContent.trim())) failures.push(`${relative}: conversation was not rendered as a chat`);
    }
    for (const link of doc.querySelectorAll('a[href]')) {
        const href = link.getAttribute('href');
        const url = new URL(href, current);
        if (url.origin !== current.origin) continue;
        const pathname = decodeURIComponent(url.pathname);
        if (base && !pathname.startsWith(base + '/')) { failures.push(`${relative}: link escapes deployment base: ${href}`); continue; }
        const local = path.join(root, pathname.slice(base.length).replace(/^\//, ''));
        const target = pathname.endsWith('/') ? path.join(local, 'index.html') : local;
        try { await fs.access(target); } catch { failures.push(`${relative}: missing ${href}`); continue; }
        if (url.hash && documents.has(target) && !documents.get(target).getElementById(decodeURIComponent(url.hash.slice(1)))) failures.push(`${relative}: missing anchor ${href}`);
    }
}
if (failures.length) { console.error([...new Set(failures)].join('\n')); process.exitCode = 1; }
else console.log(`Checked internal links, anchors, assets, and conversation structure across ${pages.length} pages.`);
