# Cache Command

The cache command manages message caching for channels to optimize search and context operations.

## Usage

```
/cache [action] [channel] [pattern]
```

## Parameters

### Required
- `action`: Action to perform (clear/status/clean)
- `channel`: Channel to manage cache for (not required for clean)

### Optional
- `pattern`: File pattern for cleanup (only for clean action)

## Examples

### Clear Channel Cache
```
/cache action:clear channel:#general
```

### Check Cache Status
```
/cache action:status channel:#team-updates
```

### Clean Old Cache Files
```
/cache action:clean pattern:*.old.json
```

## Permissions

- Requires MANAGE_MESSAGES or appropriate role
- Rate limited to prevent abuse
- Restricted to moderators by default

## Response Format

### Success (Clear)
```typescript
{
  embeds: [
    {
      title: "Cache Cleared",
      description: "Cache cleared for {channel}",
      fields: [
        {
          name: "Messages Cleared",
          value: "{count}"
        },
        {
          name: "Space Freed",
          value: "{size}"
        }
      ]
    }
  ]
}
```

### Success (Status)
```typescript
{
  embeds: [
    {
      title: "Cache Status",
      description: "Current cache status for {channel}",
      fields: [
        {
          name: "Cached Messages",
          value: "{message_count}"
        },
        {
          name: "Cache Size",
          value: "{cache_size}"
        },
        {
          name: "Last Updated",
          value: "{timestamp}"
        }
      ]
    }
  ]
}
```

### Error States
- Channel not found
- No cache exists
- Permission denied
- Invalid pattern

## Cache Management

### Cache Types
1. Message Content Cache
2. Analysis Results Cache
3. Search Index Cache
4. Context Window Cache

### Cleanup Criteria
- Age-based (older than X days)
- Size-based (exceeding limits)
- Pattern-based (matching files)
- Manual (explicit clear)

## Best Practices

1. **Cache Maintenance**
   - Regular cleanup
   - Monitor sizes
   - Validate integrity
   - Keep frequently used

2. **Performance**
   - Clear unused caches
   - Optimize storage
   - Balance retention
   - Schedule cleanups

3. **Organization**
   - Document changes
   - Track patterns
   - Monitor usage
   - Plan capacity

## Configuration

```typescript
const CACHE_CONFIG = {
  // Storage limits
  MAX_CACHE_SIZE: '1GB',
  MAX_FILES: 1000,
  MAX_AGE: '7d',
  
  // Cleanup settings
  AUTO_CLEANUP: true,
  CLEANUP_INTERVAL: '1d',
  MIN_AGE: '1h',
  
  // Performance
  COMPRESSION: true,
  BATCH_SIZE: 100,
  PARALLEL_OPS: 3,
  
  // Monitoring
  LOG_OPERATIONS: true,
  ALERT_THRESHOLD: 0.9,
  TRACK_STATS: true
};
```

For implementation details, see the [FileSystem Service Documentation](../services/FileSystemService.md).