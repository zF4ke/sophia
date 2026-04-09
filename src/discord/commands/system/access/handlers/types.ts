import type {
    ButtonInteraction,
    ModalSubmitInteraction,
    StringSelectMenuInteraction,
    UserSelectMenuInteraction,
} from "discord.js";
import type { BotClient } from "@/shared/appTypes";

export type AccessInteraction =
    | ButtonInteraction
    | StringSelectMenuInteraction
    | UserSelectMenuInteraction
    | ModalSubmitInteraction;

export type AccessHandlerContext = {
    client: BotClient;
};
