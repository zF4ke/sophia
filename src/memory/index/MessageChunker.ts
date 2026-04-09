const CHUNK_LENGTH = 500;

export class MessageChunker {
    public static split(content: string): string[] {
        if (content.length <= CHUNK_LENGTH) {
            return [content];
        }

        const chunks: string[] = [];
        let currentIndex = 0;
        while (currentIndex < content.length) {
            chunks.push(content.slice(currentIndex, currentIndex + CHUNK_LENGTH));
            currentIndex += CHUNK_LENGTH;
        }
        return chunks;
    }
}
