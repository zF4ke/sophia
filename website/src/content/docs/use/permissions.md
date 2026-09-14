---
title: Permissions and server actions
sidebar:
  order: 4
---

Sophia works only for admitted Discord accounts in enabled locations. Names in a message cannot impersonate another account. An operator grants access by verified user ID or a scoped role.

## Personal setup

Run this once in the server where you want Sophia, selecting your own account for `user`:

```text
/access enable_here:true user:@you level:destructive mode:ask
```

This enables that server and authorizes you to request all action types. Sophia reads without asking and requests approval before changes, including deletions. Your Discord account ID identifies you automatically. Other people do not gain access from seeing your public commands or panels.

Use `mode:auto` if you want authorized changes, including deletions at this level, to run without approval. Use `level:write` if you want deletions blocked altogether. You can ignore roles, expiry and individual tool rules for personal use.

`/access`, `/skills`, `/memories`, `/schedules` and task exports appear in the channel by default. Add `ephemeral:true` to a command when you want its reply visible only to you. Message visibility does not change who can operate Sophia.

## Reads and changes

| Policy | What happens |
| --- | --- |
| Read grant | Sophia can inspect eligible information. It cannot make external changes. |
| Ask, write tier | Reads run freely. A change shows a preview for the requester to approve. |
| Auto-approve, write tier | Authorized writes run without another prompt. Destructive actions remain outside the grant. |
| Destructive tier | Destructive actions may run according to Ask or Auto mode. Protected resources and explicit denies still apply. |

Rules can target a capability or tier. Deny wins; a rule cannot exceed your grant's ceiling. Approval belongs to the authenticated requester and bounded action. Different arguments require a new decision. Expiry is distinct from denial.



> **You**: Create a text channel called event-photos under Community.
>
> **Sophia**: Create #event-photos under Community? [Approve] [Deny]
>
> **You**: [Approve]
>
> **Sophia**: Created #event-photos. [Channel link]

The actual receipt includes the channel ID and mention. Sophia can also create categories, threads and roles; move or edit channels/categories; manage member roles; send or edit messages; and delete selected messages, channels or roles when authorized. It cannot bypass Discord's own role hierarchy or channel permissions.

## Shared knowledge does not erase privacy

Source access is checked during reads and publication. Memories, skills, cards, files and prior answers retain source dependencies. A private source cannot be published just because someone asks in a public channel. Revocation during work can stop subsequent access or dispatch.

For operator commands, read [access policy](../../build/access-policy/). For uncertain outcomes, read [task recovery](../tasks/).
