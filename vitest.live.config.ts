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
        include: ["tests/live/**/*.test.ts"],
        testTimeout: 120000,
        hookTimeout: 120000,
    },
});
