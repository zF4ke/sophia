import { describe, expect, it } from "vitest";
import {
    ARTIFACT_LIMITS,
    artifactExpiryTimestamp,
    buildArtifactComponents,
    buildArtifactContainer,
    validateArtifactSpec,
} from "@/discord/artifacts/ArtifactBuilder";

describe("artifact builder", () => {
    it("accepts a minimal valid spec", () => {
        const result = validateArtifactSpec({
            title: "Weekly digest",
            sections: [{ heading: "Highlights", body: "Three things happened." }],
        });

        expect(result.ok).toBe(true);
        if (result.ok) {
            expect(result.value.title).toBe("Weekly digest");
            expect(result.value.ephemeral).toBe(false);
            expect(result.value.ttlDays).toBe(0);
            expect(result.value.navigation).toBeUndefined();
        }
    });

    it("rejects empty titles, empty sections, and oversized bodies", () => {
        expect(validateArtifactSpec({ title: "", sections: [{ body: "x" }] }).ok).toBe(false);
        expect(validateArtifactSpec({ title: "t", sections: [] }).ok).toBe(false);
        expect(validateArtifactSpec({
            title: "t",
            sections: [{ body: "x".repeat(ARTIFACT_LIMITS.maxBodyChars + 1) }],
        }).ok).toBe(false);
        expect(validateArtifactSpec({
            title: "t",
            sections: [{ body: "ok" }],
            ttl_days: ARTIFACT_LIMITS.maxTtlDays + 1,
        }).ok).toBe(false);
    });

    it("rejects invalid navigation and link buttons", () => {
        expect(validateArtifactSpec({
            title: "t",
            sections: [{ body: "only one" }],
            navigation: { type: "pagination" },
        }).ok).toBe(false);

        expect(validateArtifactSpec({
            title: "t",
            sections: [{ body: "a" }, { body: "b" }],
            navigation: { type: "carousel" },
        }).ok).toBe(false);

        expect(validateArtifactSpec({
            title: "t",
            sections: [{ body: "a" }],
            link_buttons: [{ label: "nope", url: "http://insecure.example" }],
        }).ok).toBe(false);
    });

    it("recovers sections and link_buttons sent as JSON strings", () => {
        const result = validateArtifactSpec({
            title: "t",
            sections: JSON.stringify([{ body: "a" }, { body: "b" }]),
            navigation: JSON.stringify({ type: "pagination" }),
            link_buttons: JSON.stringify([{ label: "Docs", url: "https://example.com" }]),
        });

        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.value.sections).toHaveLength(2);
        expect(result.value.navigation).toEqual({ type: "pagination" });
        expect(result.value.linkButtons).toEqual([{ label: "Docs", url: "https://example.com" }]);
    });

    it("validates media URLs: direct images pass, pages fail", () => {
        const ok = validateArtifactSpec({
            title: "t",
            sections: [{ body: "a", thumbnail_url: "https://cdn.discordapp.com/avatars/1/2.png?size=256" }],
            gallery: ["https://i.imgur.com/abc.gif"],
        });
        expect(ok.ok).toBe(true);

        const page = validateArtifactSpec({
            title: "t",
            sections: [{ body: "a", thumbnail_url: "https://example.com/some/page" }],
        });
        expect(page.ok).toBe(false);
        if (!page.ok) expect(page.error).toMatch(/direct image link/i);
    });

    it("accepts files, accent color, and spoiler", () => {
        const result = validateArtifactSpec({
            title: "t",
            sections: [{ body: "a" }],
            files: ["https://cdn.discordapp.com/attachments/1/2/report.pdf"],
            accent_color: "#9aa7ff",
            spoiler: true,
        });

        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.value.files).toHaveLength(1);
        expect(result.value.accentColor).toBe(0x9aa7ff);
        expect(result.value.spoiler).toBe(true);

        const badColor = validateArtifactSpec({
            title: "t",
            sections: [{ body: "a" }],
            accent_color: "not-a-color",
        });
        expect(badColor.ok).toBe(false);
    });

    it("renders section thumbnails, galleries, and file components", () => {
        const parsed = validateArtifactSpec({
            title: "Dossiê",
            sections: [
                { heading: "Arguido", body: "Corpo.", thumbnail_url: "https://cdn.discordapp.com/avatars/1/2.png?size=256" },
                { body: "Sem imagem." },
            ],
            gallery: ["https://i.imgur.com/abc.gif"],
            files: ["https://cdn.discordapp.com/attachments/1/2/report.pdf"],
            accent_color: "#ff0000",
        });
        expect(parsed.ok).toBe(true);
        if (!parsed.ok) return;

        const built = buildArtifactComponents(parsed.value, { section: 0 }, "nonce9");
        const containerJson = (built.components[0] as unknown as { toJSON: () => any }).toJSON();
        expect(containerJson.accent_color).toBe(0xff0000);

        // First section: Section component (9) with thumbnail accessory (11).
        const sectionJson = containerJson.components.find((c: any) => c.type === 9);
        expect(sectionJson.accessory.type).toBe(11);
        expect(sectionJson.accessory.media.url).toContain("avatars/1/2.png");
        // Second section stays a plain text display.
        expect(containerJson.components.some((c: any) => c.type === 10)).toBe(true);
        // Gallery (12) and file (13) components present.
        expect(containerJson.components.some((c: any) => c.type === 12)).toBe(true);
        expect(containerJson.components.some((c: any) => c.type === 13)).toBe(true);
        expect(containerJson.components.find((c: any) => c.type === 13).file.url).toBe("attachment://report.pdf");
    });

    it("enforces the total char budget across all content", () => {
        const body = "x".repeat(1800);
        expect(validateArtifactSpec({
            title: "t",
            sections: [{ body }, { body }],
        }).ok).toBe(true);

        expect(validateArtifactSpec({
            title: "t",
            sections: [{ body }, { body }, { body }],
        }).ok).toBe(false);
    });

    it("builds per-state components with pagination and select controls", () => {
        const parsed = validateArtifactSpec({
            title: "Report",
            summary: "Short version.",
            sections: [
                { heading: "A", body: "First." },
                { heading: "B", body: "Second." },
            ],
            navigation: { type: "pagination" },
            link_buttons: [{ label: "Docs", url: "https://example.com/docs" }],
            ttl_days: 30,
        });
        expect(parsed.ok).toBe(true);
        if (!parsed.ok) return;

        const first = buildArtifactComponents(parsed.value, { section: 0 }, "nonce1");
        expect(first.pageCount).toBe(2);
        expect(first.section).toBe(0);
        expect(first.components.length).toBe(3); // container + pagination row + link row

        const last = buildArtifactComponents(parsed.value, { section: 1 }, "nonce1", { disabled: true });
        expect(last.section).toBe(1);

        const container = buildArtifactContainer(parsed.value);
        expect(container).toBeDefined();
        expect(artifactExpiryTimestamp(parsed.value.ttlDays, 1_000)).toBe(1_000 + 30 * 86_400_000);
        expect(artifactExpiryTimestamp(0, 1_000)).toBeNull();
    });
});
