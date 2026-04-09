import { DISCORD } from "@/discord/constants";

export class MessageChunker {
    public static split(message: string, limit: number = DISCORD.MESSAGE_LIMIT): string[] {
        if (message.length <= limit) {
            return [message];
        }

        const chunks: string[] = [];
        let current = "";
        const paragraphs = message.split("\n\n");

        for (const paragraph of paragraphs) {
            const candidate = current ? `${current}\n\n${paragraph}` : paragraph;
            if (candidate.length <= limit) {
                current = candidate;
                continue;
            }

            if (current) {
                chunks.push(current);
            }

            if (paragraph.length <= limit) {
                current = paragraph;
                continue;
            }

            let offset = 0;
            while (offset < paragraph.length) {
                chunks.push(paragraph.slice(offset, offset + limit));
                offset += limit;
            }
            current = "";
        }

        if (current) {
            chunks.push(current);
        }

        return chunks;
    }
}
