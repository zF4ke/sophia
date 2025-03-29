# Admin Command

The `/admin` command allows managing the bot's security system, including administrators, moderators, and command configurations.

This command is restricted to bot administrators only.

## Subcommands

### Administrator Management

#### `/admin admins list`

Lists all registered administrators, showing who added them and when.

#### `/admin admins add user:[user]`

Adds a new administrator. The user will have access to all administrative functionalities of the bot.

#### `/admin admins remove user:[user]`

Removes an existing administrator. Fixed administrators defined in the code cannot be removed.

### Moderator Management

#### `/admin moderators list` 

Lists all registered moderators, showing who added them and when.

#### `/admin moderators add user:[user]`

Adds a new moderator. The user will have access to public commands with increased rate limits.

#### `/admin moderators remove user:[user]`

Removes an existing moderator.

### Command Management

#### `/admin command visibility command:[command] public:[true/false]`

Sets whether a command is public (available to all users) or private (available to administrators only).

**Parameters:**
- `command`: Name of the command to configure
- `public`: Whether the command should be public (`true`) or private (`false`)

#### `/admin command limit command:[command] default:[limit] admin:[limit] moderator:[limit]`

Sets usage limits for a specific command.

**Parameters:**
- `command`: Name of the command to configure
- `default`: Default limit for regular users
- `admin`: (Optional) Limit for administrators
- `moderator`: (Optional) Limit for moderators

## Examples

### Add a Moderator
```
/admin moderators add user:@user
```

### Make a Command Private
```
/admin command visibility command:search public:false
```

### Set Rate Limits
```
/admin command limit command:hello default:5 admin:20 moderator:10
```

## Notes

- Changes are saved automatically and persist between bot restarts
- Rate limits are applied per user, per command, and reset after a period of time (default: 1 minute)
- Administrators can use all commands, even those configured as private