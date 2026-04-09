require("dotenv").config();

import { bootstrapRuntime } from "@/app/bootstrapRuntime";
import { createClient } from "@/app/createClient";
import { registerProcessHandlers } from "@/app/registerProcessHandlers";
import type { BotClient } from "@/shared/appTypes";

const client = createClient();
registerProcessHandlers();
void bootstrapRuntime(client);

export default client;
export type { BotClient };
