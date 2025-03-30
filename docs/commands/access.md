# Access Command

The access command manages command access permissions for roles and users.

## Usage

```
/access [action] [command] [role] [duration] [channels]
```

## Parameters

### Required
- `action`: Action to perform (grant/revoke/list)
- `command`: Command to manage access for

### Optional
- `role`: Role to grant/revoke access (required for grant/revoke)
- `duration`: Access duration (e.g., "7d", "30d")
- `channels`: Specific channels to restrict access to

## Examples

### Grant Access
```
/access action:grant command:search role:@Moderator duration:30d
```

### Revoke Access
```
/access action:revoke command:context role:@everyone
```

### List Permissions
```
/access action:list command:search
```

## Permissions

- Requires ADMINISTRATOR or appropriate role
- Cannot modify own permissions
- Cannot grant higher permissions than own

## Response Format

### Success (Grant/Revoke)
```typescript
{
  embeds: [
    {
      title: "Access Updated",
      description: "Permission changes applied",
      fields: [
        {
          name: "Command",
          value: "{command_name}"
        },
        {
          name: "Role",
          value: "{role_name}"
        },
        {
          name: "Action",
          value: "{action_type}"
        },
        {
          name: "Duration",
          value: "{duration}"
        }
      ]
    }
  ]
}
```

### Success (List)
```typescript
{
  embeds: [
    {
      title: "Command Access",
      description: "Current permissions for {command}",
      fields: [
        // List of roles and their access levels
      ]
    }
  ]
}
```

### Error States
- Invalid command
- Invalid role
- Insufficient permissions
- Invalid duration format

## Access Levels

### Role Hierarchy
1. Administrator (Full access)
2. Moderator (Limited commands)
3. User (Basic commands)
4. Restricted (Minimal access)

### Command Categories
- Administrative (access, config)
- Moderation (search, context)
- User (getmessage)
- Public (help, ping)

## Best Practices

1. **Permission Management**
   - Follow least privilege
   - Regular access reviews
   - Document changes
   - Use appropriate durations

2. **Security**
   - Verify permissions
   - Audit access changes
   - Monitor usage
   - Handle errors gracefully

3. **Organization**
   - Group related permissions
   - Clear role hierarchy
   - Channel-specific access
   - Temporary access when needed

## Configuration

```typescript
const ACCESS_CONFIG = {
  // Permission levels
  LEVELS: {
    ADMIN: 3,
    MOD: 2,
    USER: 1,
    RESTRICTED: 0
  },
  
  // Duration settings
  DEFAULT_DURATION: '30d',
  MAX_DURATION: '365d',
  MIN_DURATION: '1d',
  
  // Command categories
  CATEGORIES: {
    ADMIN: ['access', 'config'],
    MOD: ['search', 'context'],
    USER: ['getmessage'],
    PUBLIC: ['help', 'ping']
  },
  
  // Audit settings
  LOG_CHANGES: true,
  NOTIFY_UPDATES: true,
  REQUIRE_REASON: true
};
```

For implementation details, see the [Security Service Documentation](../services/SecurityService.md).