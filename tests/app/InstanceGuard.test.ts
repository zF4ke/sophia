import path from "node:path";
import { expect, it } from "vitest";
import { AppPaths } from "@/app/AppPaths";
import { acquireInstanceGuard } from "@/app/InstanceGuard";

it("rejects concurrent ownership and releases it cleanly", async () => {
    const location = path.join(AppPaths.storageRoot, "instance-test");
    const release = await acquireInstanceGuard(location);
    try { await expect(acquireInstanceGuard(location)).rejects.toThrow("exclusive ownership"); }
    finally { await release(); }
    const next = await acquireInstanceGuard(location);
    await next();
});
