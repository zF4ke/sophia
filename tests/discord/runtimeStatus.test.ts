import { expect, it } from "vitest";
import { isRuntimeStatus } from "@/shared/runtimeStatus";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { vi } from "vitest";

const status = { client: { user: { id: "sophia" } }, author: { id: "sophia" }, components: [{ components: [{ customId: "task:stop:task" }, { customId: "task:details:task" }] }] };
it("excludes Sophia's mutable task controls, but retains ordinary bot replies and other people's messages", () => {
    expect(isRuntimeStatus(status as never)).toBe(true);
    expect(isRuntimeStatus({ ...status, author: { id: "person" } } as never)).toBe(false);
    expect(isRuntimeStatus({ ...status, components: [] } as never)).toBe(false);
});
it("does not ingest a progress update as source evidence", async () => {
    const ingest = vi.spyOn(DiscordMemoryService, "ingestStoredMessage");
    try {
        await DiscordMemoryService.ingestMessage(status as never);
        expect(ingest).not.toHaveBeenCalled();
    } finally { ingest.mockRestore(); }
});
