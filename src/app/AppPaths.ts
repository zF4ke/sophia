import path from "path";

const workspaceRoot = process.cwd();

export class AppPaths {
    public static readonly workspaceRoot = workspaceRoot;
    public static readonly sourceRoot = path.join(workspaceRoot, "src");
    public static readonly resourcesRoot = path.join(workspaceRoot, "resources");
    public static readonly storageRoot = path.join(workspaceRoot, "storage");
    public static readonly docsRoot = path.join(workspaceRoot, "docs");
    public static readonly promptsRoot = path.join(this.resourcesRoot, "prompts");
    public static readonly modelProfilesPath = path.join(
        this.resourcesRoot,
        "models",
        "model-profiles.json"
    );
    public static readonly commandModulesRoot = path.join(
        this.sourceRoot,
        "discord",
        "commands"
    );
    public static readonly eventModulesRoot = path.join(
        this.sourceRoot,
        "discord",
        "events"
    );
}
