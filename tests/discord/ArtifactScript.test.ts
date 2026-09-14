import { afterEach, expect, it, vi } from "vitest";
import { ContainerSandbox } from "@/runtime/sandbox/ContainerSandbox";
import { runArtifactScript } from "@/discord/artifacts/ArtifactScript";
afterEach(() => vi.restoreAllMocks());
const input = { code: "throw new Error('must not run on host')", state: {}, user: { id: "u1", username: "User" }, values: [], customId: "x", cardId: "m1" };
it("runs artifact code only through the container boundary", async () => {
    const execute = vi.spyOn(ContainerSandbox, "execute").mockResolvedValue({ exitCode: 0, stdout: JSON.stringify({ state: { score: 1 }, reply: "Done", sends: [], logs: [], error: null }), stderr: "", files: [] });
    const result = await runArtifactScript(input);
    expect(result.state).toEqual({ score: 1 });
    expect(execute).toHaveBeenCalledWith(expect.objectContaining({ language: "javascript", code: expect.stringContaining("must not run on host") }));
});
it("preserves state and queues no sends when the container is unavailable", async () => {
    vi.spyOn(ContainerSandbox, "execute").mockRejectedValue(new Error("Docker unavailable"));
    expect(await runArtifactScript(input)).toMatchObject({ state: {}, sends: [], error: "Docker unavailable" });
});
it("rejects malformed sends returned by an untrusted script", async () => {
    vi.spyOn(ContainerSandbox, "execute").mockResolvedValue({ exitCode: 0, stdout: JSON.stringify({ state: {}, sends: [{ channelId: "arbitrary", content: "text" }], logs: [] }), stderr: "", files: [] });
    expect((await runArtifactScript(input)).sends).toEqual([]);
});
