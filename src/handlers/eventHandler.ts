import { TDiscordClient } from "..";

const ascii = require("ascii-table");
import fs from "fs";
const table = new ascii().setHeading("Events", "Status");

function loadEvents(client: TDiscordClient) {

    const folders = fs.readdirSync("./src/events");
    for (const folder of folders) {
        const files = fs
            .readdirSync(`./src/events/${folder}`)
            .filter((file: string) => file.endsWith(".js") || file.endsWith(".ts"));

        for (const file of files) {
            const event = require(`../events/${folder}/${file}`);

            if (event.rest) {
                if (event.once)
                    client.rest.once(event.name, (...args: any[]) =>
                        event.execute(...args, client)
                    );
                else
                    client.rest.on(event.name, (...args: any[]) =>
                        event.execute(...args, client)
                    );
            } else {
                if (event.once)
                    client.once(event.name, (...args) =>
                        event.execute(...args, client)
                    );
                else
                    client.on(event.name, (...args) =>
                        event.execute(...args, client)
                    );
            }
            table.addRow(file, "loaded");
            continue;
        }
    }
    return console.log(table.toString(), "\nLoaded events");
}

export { loadEvents };