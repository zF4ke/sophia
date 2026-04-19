import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "src"),
        },
    },
    test: {
        environment: "node",
        include: ["tests/**/*.test.ts"],
        exclude: ["tests/live/**/*.test.ts"],
        setupFiles: ["tests/setup/testStorage.ts"],
    },
});
