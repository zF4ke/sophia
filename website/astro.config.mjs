import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import discordExamples from './plugins/discord-examples.mjs';

export default defineConfig({
  markdown: { remarkPlugins: [discordExamples] },
  site: process.env.DOCS_SITE || 'http://localhost:4321',
  base: process.env.DOCS_BASE || '/',
  integrations: [starlight({
    title: 'Sophia',
    logo: { src: '../assets/logo.svg', alt: '' },
    favicon: '/icon.svg',
    description: 'The Sophia 5 handbook. Use the assistant, understand the runtime, and run your own installation.',
    customCss: ['./src/styles/custom.css'],
    sidebar: [
      { label: 'The handbook', link: '/' },
      { label: 'Use Sophia', items: [{ autogenerate: { directory: 'use' } }] },
      { label: 'Run your own', items: [{ autogenerate: { directory: 'setup' } }] },
      { label: 'Build Sophia', items: [{ autogenerate: { directory: 'build' } }], collapsed: true },
      { label: 'Reference', items: [{ autogenerate: { directory: 'reference' } }], collapsed: true },
    ],
  })],
});
