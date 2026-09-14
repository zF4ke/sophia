import { expect, it, vi } from "vitest";
import { interactionApprovalTransport } from "@/discord/approval/InteractionApprovalTransport";
it("keeps an ephemeral approval preview and its updates off the public channel", async () => {
    const channelSend = vi.fn();
    const followUp = vi.fn().mockResolvedValue({ id: "private-message" });
    const editMessage = vi.fn().mockResolvedValue({});
    const interaction = { ephemeral: true, channel: { send: channelSend }, followUp, webhook: { editMessage } };
    const transport = interactionApprovalTransport(interaction as never)!;
    const sent = await transport.send({ components: [], flags: 32768 });
    await sent.edit({ components: [] });
    expect(channelSend).not.toHaveBeenCalled();
    expect(followUp).toHaveBeenCalledWith(expect.objectContaining({ flags: 32832, allowedMentions: { parse: [] } }));
    expect(editMessage).toHaveBeenCalledWith("private-message", { components: [] });
});
