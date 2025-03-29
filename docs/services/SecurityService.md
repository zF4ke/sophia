# Security Service

The SecurityService is a robust system for managing permissions and command usage limits in the bot.

## Key Features

* Management of administrators and moderators
* Control of command visibility (public/private)
* Command usage rate limiting based on user roles
* Data persistence in JSON files

## Permission Hierarchy

The permission system has three main levels:

### 1. Administrators

Administrators have full access to all bot functionalities.

**Permissions:**
- Access to all commands, including private ones
- Ability to add/remove other administrators
- Ability to add/remove moderators
- Configure command visibility
- Configure command rate limits
- Increased command usage limits (2x the default)

### 2. Moderators

Moderators have elevated permissions but less than administrators.

**Permissions:**
- Access to public commands
- Intermediate command usage limit (1.5x the default)
- No access to the admin command

### 3. Regular Users

Regular users with limited access.

**Permissions:**
- Access to public commands only
- Default command usage limit

## Data Management

The service uses JSON files for persistence:

- `data/admins.json` - List of administrators
- `data/moderators.json` - List of moderators
- `data/commands_config.json` - Command configurations (visibility and limits)

## Main Methods

### User Management

- `isAdmin(userId)` - Checks if a user is an administrator
- `isModerator(userId)` - Checks if a user is a moderator
- `addAdmin(userId, addedBy)` - Adds a new administrator
- `addModerator(userId, addedBy)` - Adds a new moderator
- `removeAdmin(userId)` - Removes an administrator
- `removeModerator(userId)` - Removes a moderator
- `getAllAdmins()` - Lists all administrators
- `getAllModerators()` - Lists all moderators

### Command Management

- `setCommandVisibility(commandName, isPublic)` - Sets whether a command is public or private
- `setCommandRateLimit(commandName, defaultLimit, adminLimit, moderatorLimit)` - Sets usage limits for a specific command
- `isCommandPublic(commandName)` - Checks if a command is public
- `checkRateLimit(userId, commandName)` - Checks if a user can use a command based on their rate limit
- `getCommandRemainingUses(userId, commandName)` - Gets the remaining usage count for a command for a user

### Utilities

- `validateChannelPermissions(interaction)` - Validates if the bot has the necessary permissions in the channel
- `cleanRateLimits()` - Cleans expired rate limits

## Usage Example

```typescript
// Check if the user is an administrator
const isAdmin = SecurityService.isAdmin(userId);

// Check if the command is public or the user can use it
const isPublic = await SecurityService.isCommandPublic(commandName);
if (!isPublic && !SecurityService.isAdmin(userId)) {
  // Access denied
}

// Check rate limit
if (!await SecurityService.checkRateLimit(userId, commandName)) {
  // Rate limit exceeded
}
```

## Integration with Commands

The SecurityService is integrated with the command system through the interaction handler, automatically verifying permissions and limits for each executed command.