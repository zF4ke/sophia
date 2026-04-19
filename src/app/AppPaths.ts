import path from "path";

const workspaceRoot = process.cwd();

// Honor SOPHIA_STORAGE_ROOT so tests (and tooling) can redirect all
// mutable state away from the real `storage/` directory. Without this,
// any test that calls SettingsService.update() — or any service that
// writes through FileSystemService — silently corrupts the user's real
// settings.json and runtime DBs.
const resolvedStorageRoot = process.env.SOPHIA_STORAGE_ROOT
    ? path.resolve(process.env.SOPHIA_STORAGE_ROOT)
    : path.join(workspaceRoot, "storage");

export class AppPaths {
    public static readonly workspaceRoot = workspaceRoot;
    public static readonly sourceRoot = path.join(workspaceRoot, "src");
    public static readonly resourcesRoot = path.join(workspaceRoot, "resources");
    public static readonly storageRoot = resolvedStorageRoot;
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
