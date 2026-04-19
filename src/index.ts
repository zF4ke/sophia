require("node:process").loadEnvFile();

import { bootstrapRuntime } from "@/app/bootstrapRuntime";
import { createClient } from "@/app/createClient";
import { registerProcessHandlers } from "@/app/registerProcessHandlers";

const client = createClient();
registerProcessHandlers();
void bootstrapRuntime(client);

export default client;
