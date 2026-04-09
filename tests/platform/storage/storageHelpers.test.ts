import fs from "fs";
import path from "path";
import { afterEach, describe, expect, it } from "vitest";
import { AppPaths } from "@/app/AppPaths";
import { ensureDirectoryExists, getStorageDir, getStorageFilePath } from "@/platform/storage/directories";
import { readJsonFile, writeJsonFile } from "@/platform/storage/jsonStore";
import { readTextFile, writeTextFile } from "@/platform/storage/textStore";
import { readBinaryFile, writeBinaryFile } from "@/platform/storage/binaryStore";
import { getDirectoryFiles, getDirectorySize, listFiles } from "@/platform/storage/fileInfo";

const tempRoot = path.join(AppPaths.storageRoot, "test-storage-helpers");

afterEach(() => {
    fs.rmSync(tempRoot, { recursive: true, force: true });
});

describe("storage helpers", () => {
    it("writes and reads json, text, and binary files", () => {
        ensureDirectoryExists(tempRoot);

        const jsonPath = getStorageFilePath("sample.json", "test-storage-helpers");
        const textPath = path.join(getStorageDir("test-storage-helpers"), "sample.txt");
        const binaryPath = path.join(getStorageDir("test-storage-helpers"), "sample.bin");

        expect(writeJsonFile(jsonPath, { ok: true })).toBe(true);
        expect(writeTextFile(textPath, "hello")).toBe(true);
        expect(writeBinaryFile(binaryPath, Buffer.from("abc"))).toBe(true);

        expect(readJsonFile<{ ok: boolean }>(jsonPath)).toEqual({ ok: true });
        expect(readTextFile(textPath)).toBe("hello");
        expect(readBinaryFile(binaryPath)?.toString("utf-8")).toBe("abc");
    });

    it("lists files and reports directory info", () => {
        const dir = getStorageDir("test-storage-helpers");
        fs.writeFileSync(path.join(dir, "a.txt"), "one");
        fs.writeFileSync(path.join(dir, "b.txt"), "two");

        expect(listFiles(dir, ".txt").sort()).toEqual(["a.txt", "b.txt"]);
        expect(getDirectorySize(dir)).toBeGreaterThan(0);
        expect(getDirectoryFiles(dir)).toHaveLength(2);
    });
});
