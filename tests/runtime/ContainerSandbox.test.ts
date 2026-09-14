import { describe, expect, it } from "vitest";
import { safeWorkspacePath, sandboxDockerArgs, validateFiles } from "@/runtime/sandbox/ContainerSandbox";
describe("container boundary", () => {
    it.each(["../outside", "/absolute", "C:/secret", "a/../../b", "a\\b", "NUL", "con.txt", "a/", "a.", "a:stream"])("rejects unsafe path %s", value => {
        expect(safeWorkspacePath(value)).toBe(false);
    });
    it("accepts ordinary relative paths and rejects Windows path aliases", () => {
        expect(safeWorkspacePath("reports/Gráfico 1.png")).toBe(true);
        expect(() => validateFiles([{ path: "Report.txt", data: "" }, { path: "report.txt", data: "" }])).toThrow();
    });
    it("does not mount host paths, grant privilege or expose networking", () => {
        const args = sandboxDockerArgs("sophia-test", "sophia-sandbox:5");
        for (const flag of ["--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--memory=512m", "--pids-limit=64", "--user=65534:65534"]) expect(args).toContain(flag);
        expect(args.some(arg => /--(volume|mount|privileged|env-file)/.test(arg))).toBe(false);
    });
});
