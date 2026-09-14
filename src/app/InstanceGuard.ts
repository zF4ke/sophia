import net from "node:net";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";

/** An OS-owned listener prevents two processes from claiming the same state store.
 * It carries no application data and disappears when the process exits. */
export async function acquireInstanceGuard(storageRoot: string): Promise<() => Promise<void>> {
    fs.mkdirSync(storageRoot, { recursive: true });
    const canonical = fs.realpathSync(path.resolve(storageRoot));
    const hash = createHash("sha256").update(process.platform === "win32" ? canonical.toLowerCase() : canonical).digest("hex");
    const address = process.platform === "win32"
        ? { path: `\\\\.\\pipe\\sophia-${hash}`, exclusive: true }
        : { host: "127.0.0.1", port: 20000 + Number.parseInt(hash.slice(0, 8), 16) % 40000, exclusive: true };
    const server = net.createServer(socket => socket.destroy());
    await new Promise<void>((resolve, reject) => {
        server.once("error", error => reject(new Error(`Sophia cannot acquire exclusive ownership of ${canonical}: ${error.message}. Another process may already be using this store.`)));
        server.listen(address, resolve);
    });
    server.unref();
    return () => new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
}
