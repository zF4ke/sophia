import { ADMIN_IDS } from "@/security/adminIds";
import { SecurityConfigStore } from "@/security/storage/SecurityConfigStore";
import type { AdminUser, ModeratorUser } from "@/security/types";

export class AdminDirectory {
    private admins: AdminUser[] = [];
    private moderators: ModeratorUser[] = [];

    public load(): void {
        this.admins = SecurityConfigStore.loadAdmins() || [];
        this.moderators = SecurityConfigStore.loadModerators() || [];

        for (const adminId of ADMIN_IDS) {
            if (!this.admins.some((admin) => admin.userId === adminId)) {
                this.admins.push({
                    userId: adminId,
                    addedBy: "system",
                    addedAt: Date.now(),
                    permissions: ["*"],
                });
            }
        }

        SecurityConfigStore.saveAdmins(this.admins);
        if (!SecurityConfigStore.loadModerators()) {
            SecurityConfigStore.saveModerators(this.moderators);
        }
    }

    public isAdmin(userId: string): boolean {
        return this.admins.some((admin) => admin.userId === userId);
    }

    public isModerator(userId: string): boolean {
        return this.isAdmin(userId) || this.moderators.some((mod) => mod.userId === userId);
    }

    public addAdmin(userId: string, addedBy: string, permissions: string[] = ["*"]): boolean {
        if (this.isAdmin(userId)) {
            return false;
        }

        this.admins.push({
            userId,
            addedBy,
            addedAt: Date.now(),
            permissions,
        });
        SecurityConfigStore.saveAdmins(this.admins);
        return true;
    }

    public removeAdmin(userId: string): boolean {
        if (!this.isAdmin(userId) || ADMIN_IDS.includes(userId)) {
            return false;
        }

        const previousLength = this.admins.length;
        this.admins = this.admins.filter((admin) => admin.userId !== userId);
        const changed = this.admins.length !== previousLength;
        if (changed) {
            SecurityConfigStore.saveAdmins(this.admins);
        }
        return changed;
    }

    public addModerator(
        userId: string,
        addedBy: string,
        permissions: string[] = ["moderate"]
    ): boolean {
        if (this.isModerator(userId)) {
            return false;
        }

        this.moderators.push({
            userId,
            addedBy,
            addedAt: Date.now(),
            permissions,
        });
        SecurityConfigStore.saveModerators(this.moderators);
        return true;
    }

    public removeModerator(userId: string): boolean {
        const previousLength = this.moderators.length;
        this.moderators = this.moderators.filter((mod) => mod.userId !== userId);
        const changed = this.moderators.length !== previousLength;
        if (changed) {
            SecurityConfigStore.saveModerators(this.moderators);
        }
        return changed;
    }

    public getAllAdmins(): AdminUser[] {
        return [...this.admins];
    }

    public getAllModerators(): ModeratorUser[] {
        return [...this.moderators];
    }
}
