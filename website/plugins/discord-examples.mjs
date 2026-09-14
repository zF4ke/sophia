const text = value => ({ type: 'text', value });
const element = (tag, className, children, props = {}) => ({ type: 'paragraph', data: { hName: tag, hProperties: { className, ...props } }, children });

/** Render explicitly authored conversation examples without raw HTML or client JavaScript. */
export default function discordExamples() {
  return tree => {
    function walk(node) {
      if (node.type === 'blockquote' && node.children?.length && node.children.every(p => p.type === 'paragraph' && p.children?.[0]?.type === 'strong' && /^(You|Sophia)/.test(p.children[0].children?.[0]?.value ?? ''))) {
        const messages = node.children.map(p => {
          const label = p.children[0].children.map(c => c.value ?? '').join('');
          const bot = label.startsWith('Sophia');
          const [author, ...timing] = label.split(',');
          const content = structuredClone(p.children.slice(1));
          if (content[0]?.type === 'text') content[0].value = content[0].value.replace(/^\s*[·:]?\s*/, '');
          for (const child of content) if (child.type === 'text') child.value = child.value.replace(/\s·\s/g, ' / ');
          const body = content.flatMap(child => {
            if (child.type !== 'text') return [child];
            return child.value.split(/(\[(?:Approve|Deny|Stop|Details|Next|Previous|Files|Message link|Channel link|Native Discord poll|Schedule details|[^\]]+\.(?:csv|mp4|png|pdf))\])/g).filter(Boolean).map(value => {
              if (!value.startsWith('[')) return text(value);
              const caption = value.slice(1, -1);
              return element('span', /\.(csv|mp4|png|pdf)$|^Files$/.test(caption) ? 'chat-attachment' : 'chat-control', [text(caption)], { 'aria-label': `Simulated ${caption}` });
            });
          });
          return element('article', `chat-message ${bot ? 'is-sophia' : 'is-user'}`, [
            element('div', 'chat-avatar', [text(bot ? 'S' : 'Y')], { 'aria-hidden': 'true' }),
            element('div', 'chat-message-main', [
              element('div', 'chat-author-row', [element('strong', 'chat-author', [text(author)]), ...(bot ? [element('span', 'chat-badge', [text('APP')])] : []), ...(timing.length ? [element('span', 'chat-time', [text(timing.join(',').trim())])] : [])]),
              element('div', 'chat-message-body', body),
            ]),
          ]);
        });
        node.type = 'paragraph';
        node.data = { hName: 'section', hProperties: { className: 'discord-example', 'aria-label': 'Simulated Discord conversation' } };
        node.children = [element('div', 'chat-channel', [element('span', 'chat-channel-name', [text('# sophia')]), element('span', 'chat-example-label', [text('Simulated conversation')])]), ...messages];
        return;
      }
      node.children?.forEach(walk);
    }
    walk(tree);
  };
}
