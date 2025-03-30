# SecurityService

The SecurityService manages command access control, permissions, and rate limiting in Sophia3.

## Core Features

### Access Control
- Role-based permissions
- User-level access
- Command restrictions
- Channel limitations

### Rate Limiting
- Command-specific limits
- User cooldowns
- Server-wide quotas
- Burst protection

## Method Reference

### validateAccess
```typescript
static async validateAccess(
  interaction: CommandInteraction,
  command: string,
  options?: AccessOptions
): Promise<boolean>
```

Validates user access to a command.

#### Parameters:
- `interaction`: Command interaction
- `command`: Command name
- `options`: Validation options
  - `requireModerator`: Require moderator role
  - `allowBots`: Allow bot users
  - `channels`: Allowed channels

#### Returns:
Boolean indicating access permission

### updateRoleAccess
```typescript
static async updateRoleAccess(
  role: Role,
  command: string,
  options: RoleAccessOptions
): Promise<void>
```

Updates role-based command access.

#### Parameters:
- `role`: Discord role
- `command`: Command name
- `options`: Access configuration
  - `grant`: Grant/revoke access
  - `expires`: Access expiration
  - `restrictions`: Usage restrictions

### checkRateLimit
```typescript
static async checkRateLimit(
  user: User,
  command: string,
  options?: RateLimitOptions
): Promise<RateLimitResult>
```

Checks and updates rate limits.

#### Parameters:
- `user`: Discord user
- `command`: Command name
- `options`: Rate limit settings
  - `maxUses`: Maximum uses
  - `window`: Time window
  - `cooldown`: Cooldown period

#### Returns:
Rate limit status and remaining quota

### validatePermissions
```typescript
static validatePermissions(
  member: GuildMember,
  permissions: PermissionResolvable[]
): boolean
```

Validates Discord permissions.

#### Parameters:
- `member`: Guild member
- `permissions`: Required permissions

#### Returns:
Boolean indicating permission status

## Integration Examples

### Basic Access Check
```typescript
if (!await SecurityService.validateAccess(interaction, 'search')) {
  throw new CommandError('Insufficient permissions');
}

// Proceed with command execution
await executeSearchCommand(interaction);
```

### Role Management
```typescript
// Grant search access to moderator role
await SecurityService.updateRoleAccess(modRole, 'search', {
  grant: true,
  expires: '7d',
  restrictions: {
    channels: ['general', 'support'],
    maxUses: 100
  }
});
```

### Rate Limiting
```typescript
const rateLimit = await SecurityService.checkRateLimit(user, 'search', {
  maxUses: 10,
  window: '1h',
  cooldown: '1m'
});

if (!rateLimit.allowed) {
  throw new RateLimitError(rateLimit.remainingTime);
}
```

## Error Handling

### Access Errors
```typescript
try {
  await SecurityService.validateAccess(interaction, command);
} catch (error) {
  if (error instanceof AccessDeniedError) {
    await interaction.reply({
      content: 'You do not have permission to use this command.',
      ephemeral: true
    });
    return;
  }
  throw error;
}
```

### Rate Limit Handling
```typescript
try {
  const limit = await SecurityService.checkRateLimit(user, command);
  if (!limit.allowed) {
    const waitTime = formatTime(limit.remainingTime);
    await interaction.reply({
      content: `Please wait ${waitTime} before using this command again.`,
      ephemeral: true
    });
    return;
  }
} catch (error) {
  console.error('Rate limit check failed:', error);
  // Proceed with default limits
}
```

## Best Practices

1. **Access Control**
   - Implement least privilege
   - Use role hierarchies
   - Regular access reviews

2. **Rate Limiting**
   - Appropriate limits per command
   - Progressive cooldowns
   - User notification

3. **Error Handling**
   - Clear error messages
   - Proper logging
   - Graceful degradation

## Configuration

```typescript
const SECURITY_CONFIG = {
  // Default access settings
  defaults: {
    requireModerator: false,
    allowBots: false,
    defaultExpiry: '30d'
  },
  
  // Rate limits
  rateLimits: {
    search: {
      maxUses: 10,
      window: '1h',
      cooldown: '1m'
    },
    context: {
      maxUses: 20,
      window: '1h',
      cooldown: '30s'
    }
  },
  
  // Permission levels
  permissionLevels: {
    USER: 0,
    MODERATOR: 1,
    ADMIN: 2,
    OWNER: 3
  }
};
```

For implementation examples, see the [Examples Guide](../guides/Examples.md).