import { TDiscordClient } from "..";

const ascii = require("ascii-table");
import fs from "fs";
const table = new ascii().setHeading("Commands", "Status");

function loadCommands(client: TDiscordClient) {
    let commandsArray: any[] = [];

    const commandsFolder = fs.readdirSync("./src/commands");
    for (const folder of commandsFolder) {
        const commandFiles = fs
            .readdirSync(`./src/commands/${folder}`)
            .filter((file: string) => file.endsWith(".js") || file.endsWith(".ts"));

        for (const file of commandFiles) {
            const commandFile = require(`../commands/${folder}/${file}`);

            const properties = { folder, ...commandFile };

            client.commands.set(commandFile.data.name, properties);

            commandsArray.push(commandFile.data.toJSON());

            table.addRow(file, "loaded");
            continue;
        }
    }

    if (client.application) {
        client.application.commands.set(commandsArray);
    }

    return console.log(table.toString(), "\nLoaded Commands");
}

export { loadCommands };