import { expect, it } from "vitest";
import { normalizeDiscordIdentifiers } from "@/shared/discordIdentifiers";

it("unwraps exact mentions without inventing an ID from malformed text", () => {
    const args = { channel_id: "<#123456789012345678>", member_id: "<@!987654321098765432>",
        role_id: "12x3456789012345678", message_ids: ["<@123456789012345678>", "123456789012345678 corrupted"], content: "<#123456789012345678>" };
    normalizeDiscordIdentifiers(args);
    expect(args).toEqual({ channel_id: "123456789012345678", member_id: "987654321098765432",
        role_id: "12x3456789012345678", message_ids: ["123456789012345678", "123456789012345678 corrupted"], content: "<#123456789012345678>" });
});
