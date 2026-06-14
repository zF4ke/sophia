/**
 * Bootstrap super-admin Discord user IDs.
 *
 * These accounts are granted permanent `["*"]` permissions on startup and
 * cannot be removed via `/access`. They exist so a fresh deployment has at
 * least one trusted owner before any access config is written to storage.
 *
 * Provide them via the `BOOTSTRAP_ADMIN_IDS` environment variable as a
 * comma-separated list of Discord user IDs, e.g.:
 *
 *   BOOTSTRAP_ADMIN_IDS=123456789012345678,234567890123456789
 *
 * Defaults to an empty list when unset. When empty, configure admins at
 * runtime with the `/access` command.
 */
export const ADMIN_IDS: string[] = (process.env.BOOTSTRAP_ADMIN_IDS ?? "")
    .split(",")
    .map((id) => id.trim())
    .filter((id) => id.length > 0);
